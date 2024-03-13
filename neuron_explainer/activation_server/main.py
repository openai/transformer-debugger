"""Starts the activation server. Methods on the server are defined in separate files."""

import datetime
import os
import re
import signal

import aiohttp
import boostedblob as bbb
import fire
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from neuron_explainer.activation_server.explainer_routes import (
    AttentionExplainAndScoreMethodId,
    NeuronExplainAndScoreMethodId,
    define_explainer_routes,
)
from neuron_explainer.activation_server.inference_routes import define_inference_routes
from neuron_explainer.activation_server.interactive_model import InteractiveModel
from neuron_explainer.activation_server.read_routes import define_read_routes
from neuron_explainer.activation_server.requests_and_responses import GroupId
from neuron_explainer.models.autoencoder_context import AutoencoderContext  # noqa: F401
from neuron_explainer.models.autoencoder_context import MultiAutoencoderContext
from neuron_explainer.models.model_context import StandardModelContext, get_default_device
from neuron_explainer.models.model_registry import make_autoencoder_context


def main(
    host_name: str = "localhost",
    port: int = 80,
    model_name: str = "gpt2-small",
    mlp_autoencoder_name: str | None = None,
    attn_autoencoder_name: str | None = None,
    run_model: bool = True,
    neuron_method: str = "baseline",
    attention_head_method: str = "baseline",
    cuda_memory_debugging: bool = False,
) -> None:
    neuron_method_id = NeuronExplainAndScoreMethodId.from_string(neuron_method)
    attention_head_method_id = AttentionExplainAndScoreMethodId.from_string(attention_head_method)

    def custom_generate_unique_id(route: APIRoute) -> str:
        return f"{route.tags[0]}-{route.name}"

    app = FastAPI(generate_unique_id_function=custom_generate_unique_id)

    allow_origin_regex_str = r"https?://localhost(:\d+)?$"
    allow_origin_regex = re.compile(allow_origin_regex_str)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=allow_origin_regex_str,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # We don't just want to disable CORS for successful responses: we also want to do it for error
    # responses, which FastAPI's middleware doesn't cover. This allows the client to see helpful
    # information like the HTTP status code, which is otherwise hidden from it. To do this, we add
    # two exception handlers. It's possible we could just get away with the first one, but GPT-4
    # thought it was good to include both and who am I to disagree?
    def add_access_control_headers(request: Request, response: JSONResponse) -> JSONResponse:
        origin = request.headers.get("origin")
        # This logic does something similar to what the standard CORSMiddleware does. You can't
        # use a regex in the actual response header, but you can run the regex on the server and
        # then choose to include the header if it matches the origin.
        if origin and allow_origin_regex.fullmatch(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        print("************** Handling an unhandled exception ***********************")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception details: {exc}")
        response = add_access_control_headers(
            request,
            JSONResponse(status_code=500, content={"message": "Unhandled server exception"}),
        )

        # Check if this exception is a cuda OOM, which is unrecoverable. If it is, we should kill
        # the server.
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            print("***** Killing server due to cuda OOM *****")
            # Use SIGKILL so the return code of the top-level process is *not* 0.
            os.kill(os.getpid(), signal.SIGKILL)

        return response

    @app.exception_handler(StarletteHTTPException)
    async def handle_starlette_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        return add_access_control_headers(
            request, JSONResponse(status_code=exc.status_code, content={"message": exc.detail})
        )

    @app.get("/", tags=["hello_world"])
    def read_root() -> dict[str, str]:
        return {"Hello": "World"}

    # The FastAPI client code generation setup only generates TypeScript classes for types
    # referenced from top-level endpoints. In some cases we want to share a type across client and
    # server that isn't referenced in this way. For example, GroupId is used in requests, but only
    # as a key in a dictionary, and the generated TypeScript for dictionaries treats enum values as
    # strings, so GroupId isn't referenced in the generated TypeScript. To work around this, we
    # define a dummy endpoint that references GroupId, which causes the client code generation to
    # generate a TypeScript class for it. The same trick can be used for other types in the future.
    @app.get("/force_client_code_generation", tags=["hello_world"])
    def force_client_code_generation(group_id: GroupId) -> None:
        return None

    @app.get("/dump_memory_snapshot", tags=["memory"])
    def dump_memory_snapshot() -> str:
        if not cuda_memory_debugging:
            raise HTTPException(
                status_code=400,
                detail="The cuda_memory_debugging flag must be set to dump a memory snapshot",
            )
        formatted_time = datetime.datetime.now().strftime("%H%M%S")
        filename = f"torch_memory_snapshot_{formatted_time}.pkl"
        torch.cuda.memory._dump_snapshot(filename)
        return f"Dumped cuda memory snapshot to {filename}"

    @app.on_event("startup")
    async def setup_boostedblob() -> None:
        assert bbb.globals.config.session is None
        bbb.globals.set_event_loop_exception_handler()
        connector = aiohttp.TCPConnector(limit=0)
        bbb.globals.config.session = aiohttp.ClientSession(connector=connector)

    if run_model:
        if cuda_memory_debugging:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        device = get_default_device()
        standard_model_context = StandardModelContext(model_name, device=device)
        if mlp_autoencoder_name is not None or attn_autoencoder_name is not None:
            autoencoder_context_list = [
                make_autoencoder_context(
                    model_name=model_name,
                    autoencoder_name=autoencoder_name,
                    device=device,
                    omit_dead_latents=True,
                )
                for autoencoder_name in [mlp_autoencoder_name, attn_autoencoder_name]
                if autoencoder_name is not None
            ]
            multi_autoencoder_context = MultiAutoencoderContext.from_autoencoder_context_list(
                autoencoder_context_list
            )
            multi_autoencoder_context.warmup()
            model = InteractiveModel.from_standard_model_context_and_autoencoder_context(
                standard_model_context, multi_autoencoder_context
            )

        else:
            model = InteractiveModel.from_standard_model_context(standard_model_context)

    else:
        model = None

    define_read_routes(app)
    define_explainer_routes(
        app=app,
        neuron_method_id=neuron_method_id,
        attention_head_method_id=attention_head_method_id,
    )
    define_inference_routes(
        app=app,
        model=model,
        mlp_autoencoder_name=mlp_autoencoder_name,
        attn_autoencoder_name=attn_autoencoder_name,
    )

    # TODO(sbills): Make reload=True work. We need to pass something like "main:app" as a string
    # rather than passing a FastAPI object directly.
    uvicorn.run(app, host=host_name, port=port)


if __name__ == "__main__":
    fire.Fire(main)


"""
For local testing without running a subject model:
python neuron_explainer/activation_server/main.py --run_model False --port 8000
"""

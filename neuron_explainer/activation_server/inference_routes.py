"""Routes / endpoints related to performing inference on the subject model."""

from fastapi import FastAPI, HTTPException

from neuron_explainer.activation_server.interactive_model import InteractiveModel
from neuron_explainer.activation_server.requests_and_responses import (
    BatchedRequest,
    BatchedResponse,
    BatchedTdbRequest,
    DerivedAttentionScalarsRequest,
    DerivedAttentionScalarsResponse,
    DerivedScalarsRequest,
    DerivedScalarsResponse,
    ModelInfoResponse,
    MultipleTopKDerivedScalarsRequest,
    MultipleTopKDerivedScalarsResponse,
)


def define_inference_routes(
    app: FastAPI,
    model: InteractiveModel | None,
    mlp_autoencoder_name: str | None,
    attn_autoencoder_name: str | None,
) -> None:
    def assert_model() -> None:
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Inference model not running. Restart the activation server with run_model=True to use inference endpoints.",
            )

    @app.post("/derived_scalars", response_model=DerivedScalarsResponse, tags=["inference"])
    async def derived_scalars(request: DerivedScalarsRequest) -> DerivedScalarsResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return await model.get_derived_scalars(request)

    @app.post(
        "/derived_attention_scalars",
        response_model=DerivedAttentionScalarsResponse,
        tags=["inference"],
    )
    async def derived_attention_scalars(
        request: DerivedAttentionScalarsRequest,
    ) -> DerivedAttentionScalarsResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return await model.get_derived_attention_scalars(request)

    @app.post(
        "/multiple_top_k_derived_scalars",
        response_model=MultipleTopKDerivedScalarsResponse,
        tags=["inference"],
    )
    async def multiple_top_k_derived_scalars(
        request: MultipleTopKDerivedScalarsRequest,
    ) -> MultipleTopKDerivedScalarsResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return await model.get_multiple_top_k_derived_scalars(request)

    @app.post("/batched", response_model=BatchedResponse, tags=["inference"])
    async def batched(request: BatchedRequest) -> BatchedResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return await model.handle_batched_request(request)

    @app.post("/batched_tdb", response_model=BatchedResponse, tags=["inference"])
    async def batched_tdb(request: BatchedTdbRequest) -> BatchedResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return await model.handle_batched_tdb_request(request)

    @app.post("/model_info", response_model=ModelInfoResponse, tags=["inference"])
    def model_info() -> ModelInfoResponse:
        assert_model()
        assert model is not None  # redundant; needed for mypy
        return model.get_model_info(
            mlp_autoencoder_name=mlp_autoencoder_name, attn_autoencoder_name=attn_autoencoder_name
        )

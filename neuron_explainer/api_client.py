import asyncio
import contextlib
import os
import random
import traceback
from functools import wraps
from typing import Any

import httpx
import orjson

API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY, "Please set the OPENAI_API_KEY environment variable"
API_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + API_KEY,
}
BASE_API_URL = "https://api.openai.com/v1"


def async_exponential_backoff(retry_on=lambda err: True):
    """
    Returns a decorator which retries the wrapped function as long as the specified retry_on
    function returns True for the exception, applying exponential backoff with jitter after
    failures, up to a retry limit.
    """
    init_delay_s = 1
    max_delay_s = 10
    backoff_multiplier = 2.0
    jitter = 0.2

    def decorate(f):
        @wraps(f)
        async def f_retry(self, *args, **kwargs):
            max_tries = 200
            delay_s = init_delay_s
            for i in range(max_tries):
                try:
                    return await f(self, *args, **kwargs)
                except Exception as err:
                    if i == max_tries - 1:
                        print(f"Exceeded max tries ({max_tries}) on HTTP request")
                        raise
                    if not retry_on(err):
                        print("Unretryable error on HTTP request")
                        raise
                    jittered_delay = random.uniform(delay_s * (1 - jitter), delay_s * (1 + jitter))
                    await asyncio.sleep(jittered_delay)
                    delay_s = min(delay_s * backoff_multiplier, max_delay_s)

        return f_retry

    return decorate


def is_api_error(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        response = err.response
        error_data = response.json().get("error", {})
        error_message = error_data.get("message")
        if response.status_code in [400, 404, 415]:
            if error_data.get("type") == "idempotency_error":
                print(f"Retrying after idempotency error: {error_message} ({response.url})")
                return True
            else:
                # Invalid request
                return False
        else:
            print(f"Retrying after API error: {error_message} ({response.url})")
            return True

    elif isinstance(err, httpx.ConnectError):
        print(f"Retrying after connection error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.TimeoutException):
        print(f"Retrying after a timeout error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.ReadError):
        print(f"Retrying after a read error... ({err.request.url})")
        return True

    print(f"Retrying after an unexpected error: {repr(err)}")
    traceback.print_tb(err.__traceback__)
    return True


class ApiClient:
    """
    Performs inference using the OpenAI API. Supports response caching and concurrency limits."

    Cache is useful for rerunning code in jupyter notebooks without repeating identical requests
    """

    def __init__(
        self,
        model_name: str,
        max_concurrent: int | None = None,
        # If set, no more than this number of HTTP requests will be made concurrently.
        cache: bool = False,
    ):
        self.model_name = model_name

        if max_concurrent is not None:
            self.concurrency_check: asyncio.Semaphore | None = asyncio.Semaphore(max_concurrent)
        else:
            self.concurrency_check = None
        if cache:
            self.cache = {}
        else:
            self.cache = None

    @async_exponential_backoff(retry_on=is_api_error)
    async def async_generate(self, timeout: int | None = None, **kwargs: Any) -> dict:
        if self.cache is not None:
            key = orjson.dumps(kwargs)
            if key in self.cache:
                return self.cache[key]
        async with contextlib.AsyncExitStack() as stack:
            if self.concurrency_check is not None:
                await stack.enter_async_context(self.concurrency_check)
            http_client = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout))
            # If the request has a "messages" key, it should be sent to the /chat/completions
            # endpoint. Otherwise, it should be sent to the /completions endpoint.
            url = BASE_API_URL + ("/chat/completions" if "messages" in kwargs else "/completions")
            kwargs["model"] = self.model_name
            response = await http_client.post(url, headers=API_HTTP_HEADERS, json=kwargs)
        # Response json has useful information but the exception doesn't include it, so print it out then reraise
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.json())
            raise e
        if self.cache is not None:
            self.cache[key] = response.json()
        return response.json()


if __name__ == "__main__":

    async def main() -> None:
        client = ApiClient(model_name="gpt-3.5-turbo", max_concurrent=1)
        print(
            await client.async_generate(prompt="Why did the chicken cross the road?", max_tokens=9)
        )

    asyncio.run(main())

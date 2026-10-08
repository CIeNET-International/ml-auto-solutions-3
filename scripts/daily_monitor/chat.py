# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Posts the Chat draft through an incoming webhook (moved from the DAG).

Off unless POST=true. New Chat webhooks at Google need a policy exception
(go/webhook-security-decision); until then the Jetski skill posts the draft
after review.
"""

from typing import Any, Callable, Sequence

WEBHOOK_TIMEOUT_S = 30


def post_messages(
    url: str,
    messages: Sequence[str],
    report_date: str,
    post: Callable[..., Any],
) -> list[dict[str, Any]]:
  """Posts each message into one thread per report date.

  Args:
    url: The webhook URL. It holds the webhook key, so it is never logged.
    messages: The draft's messages, in order.
    report_date: YYYY-MM-DD, used as the thread key.
    post: requests.post, or a fake in tests.

  Returns:
    [{name, thread}] of the posted messages.
  """
  posted = []
  for message in messages:
    try:
      response = post(
          url,
          params={
              "threadKey": f"orbax-daily-monitor-{report_date}",
              "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
          },
          json={"text": message},
          timeout=WEBHOOK_TIMEOUT_S,
      )
    except Exception as e:  # pylint: disable=broad-except
      # The exception text contains the URL, which holds the webhook key.
      raise RuntimeError(
          f"Chat webhook call failed: {type(e).__name__}"
      ) from None
    if response.status_code != 200:
      raise RuntimeError(
          f"Chat webhook returned HTTP {response.status_code} after "
          f"{len(posted)} of {len(messages)} messages"
      )
    reply = response.json()
    posted.append(
        {
            "name": reply.get("name"),
            "thread": (reply.get("thread") or {}).get("name"),
        }
    )
  return posted

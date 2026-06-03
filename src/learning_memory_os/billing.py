"""Stripe billing for Memex Pro — checkout + webhook entitlement.

Activates when env is set; degrades gracefully otherwise:
  STRIPE_SECRET_KEY     create real Checkout Sessions
  STRIPE_PRICE_ID       the recurring price for Pro
  STRIPE_WEBHOOK_SECRET verify webhook signatures
  LMOS_CHECKOUT_URL     fallback: a Stripe Payment Link (no SDK/keys needed)
  LMOS_PUBLIC_URL       base url for success/cancel redirects (default localhost)

Money path: checkout carries client_reference_id=username; the webhook
(checkout.session.completed) flips that user's is_pro to true.
"""
import os


def _public_url() -> str:
    return os.environ.get("LMOS_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def checkout_url(username: str) -> str | None:
    """Return a URL to send the user to pay, or None if billing isn't configured."""
    secret = os.environ.get("STRIPE_SECRET_KEY")
    price = os.environ.get("STRIPE_PRICE_ID")
    if secret and price:
        try:
            import stripe
            stripe.api_key = secret
            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{"price": price, "quantity": 1}],
                client_reference_id=username,
                success_url=f"{_public_url()}/?upgraded=1#/readiness",
                cancel_url=f"{_public_url()}/#/readiness",
                allow_promotion_codes=True,
            )
            return session.url
        except Exception:
            pass  # fall through to payment link / None
    link = os.environ.get("LMOS_CHECKOUT_URL")
    if link:
        sep = "&" if "?" in link else "?"
        return f"{link}{sep}client_reference_id={username}"
    return None


def parse_webhook(payload: bytes, sig_header: str | None) -> dict | None:
    """Verify + parse a Stripe webhook; return the event dict, or None if invalid."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    try:
        import stripe
        if secret and sig_header:
            return stripe.Webhook.construct_event(payload, sig_header, secret)
        # No signing secret configured: best-effort parse (dev only).
        import json
        return json.loads(payload)
    except Exception:
        return None


def username_from_event(event: dict) -> str | None:
    """Extract the paying user's username (client_reference_id) from a completed
    checkout/subscription event."""
    if not event or event.get("type") not in (
        "checkout.session.completed", "checkout.session.async_payment_succeeded"
    ):
        return None
    obj = (event.get("data") or {}).get("object") or {}
    return obj.get("client_reference_id")

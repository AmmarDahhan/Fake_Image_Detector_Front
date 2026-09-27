"""Service layer.

Contains the model inference seam (``model_service``). Nothing in this
package may be imported by ``app.schemas`` or ``app.api.schemas`` in a way
that would leak framework types into the API contract.
"""

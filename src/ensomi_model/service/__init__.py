"""The model service: a websocket that serves the released Ensomi R2 chart generator to the Ensomi app.

``python -m ensomi_model.service --model-dir <dir>`` loads ``system.System`` once and speaks the complete-chart messages
of EnsomiProtocol (``chart.proto``) as binary ``Envelope`` frames: ``mapping`` turns a request into ``System.make``
arguments and its output into a ``Chart``, ``server`` is the socket and the one-generation-at-a-time queue. Needs the
``service`` extra.
"""

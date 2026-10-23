Time-Series Forecaster Generator
================================

The ``TimeSeriesForecasterGenerator`` turns a
:doc:`B-UML Neural Network model <../buml_language/model_types/nn>` into a runnable **time-series
forecaster**: one Python module that trains the network on a series, forecasts its next values,
saves and reloads the trained weights, and scores itself against the seasonal naive forecast.

The :doc:`PyTorch generator <pytorch>` writes a network and, when the model has a ``Dataset``, a
training script for images or a CSV with one target column. A forecaster needs more: windows over
a series, transforms and their inverse, seeded training, a prediction function and a backtest. This
generator adds them around the same network code: it subclasses ``PytorchGenerator`` and reuses
its templates, so the network is written exactly as the PyTorch generator writes it.

Nothing in it is about a particular dataset: the series is given at run time, and the pipeline is
a ``ForecastConfig``.

Usage
-----

.. code-block:: python

    from besser.BUML.metamodel.nn import NN, Configuration, DropoutLayer, LinearLayer
    from besser.generators.nn.forecaster import ForecastConfig, TimeSeriesForecasterGenerator

    # 24 past months plus the sine and cosine of the month in, the next 12 months out
    model = NN(name="mlp")
    model.add_layer(LinearLayer(name="hidden", in_features=26, out_features=32, actv_func="relu"))
    model.add_layer(DropoutLayer(name="dropout", rate=0.1))
    model.add_layer(LinearLayer(name="output", in_features=32, out_features=12))
    model.add_configuration(Configuration(batch_size=32, epochs=60, learning_rate=0.003,
                                          optimizer="adamW", weight_decay=0.01,
                                          loss_function="mse", metrics=["mae"]))

    config = ForecastConfig(window=24, horizon=12, season=12, seeds=(0, 1, 2))
    TimeSeriesForecasterGenerator(model, config, output_dir="output").generate()
    # -> output/forecaster_mlp.py

Then, with ``torch`` and ``numpy`` installed:

.. code-block:: python

    import forecaster_mlp as fc

    forecaster = fc.fit(values)                    # one series (a list of numbers) ...
    forecaster = fc.fit([series_a, series_b])      # ... or several: one "global" model
    fc.save(forecaster, "mlp.pt")
    next_year = fc.load("mlp.pt").predict(values, steps=12)

    forecasts = fc.backtest(values, origins=range(len(values) - 36, len(values) - 12))
    fc.mase(values, forecasts)                     # {horizon: MASE}, 1.0 = the seasonal naive

or from the command line, on a CSV with a column of values and optionally a period column:
``YYYY-MM`` gives the position in the season of a monthly series (``season=12``), ``YYYY-Qn`` that
of a quarterly one (``season=4``); otherwise the series is taken to start at position 0, and the
command says so.

.. code-block:: console

    python forecaster_mlp.py series.csv --steps 12 --backtest 20 --save mlp.pt

Parameters
----------

``TimeSeriesForecasterGenerator(model, config=None, output_dir=None, regenerate="", channel_last=False)``

- **model**: the B-UML ``NN``. It needs a ``Configuration`` (epochs, batch size, optimizer,
  learning rate, weight decay) with the ``mse`` loss. ``train_data`` and ``test_data`` are not used.
- **config**: the pipeline, a ``ForecastConfig`` (below); the defaults when omitted.
- **output_dir**: where ``forecaster_<model name>.py`` is written (default ``<cwd>/output``).
- **regenerate**: the command that regenerates the module, quoted (escaped) in its docstring.
- **channel_last**: as for the PyTorch generator. A first ``Conv1D`` marked ``permute_in`` then
  reads the window as (batch, window, channels), like a recurrent layer.

``ForecastConfig``:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Default
     - Meaning
   * - ``window``
     - ``24``
     - past values the network reads
   * - ``horizon``
     - ``12``
     - future values it returns: the size of its last layer (one network for every horizon)
   * - ``season``
     - ``12``
     - the period of the series (12 monthly, 4 quarterly, 7 daily)
   * - ``log``
     - ``True``
     - forecast the logarithm when every value of the training series is positive (otherwise the
       levels, with a warning). The trained forecaster remembers the decision: one trained on the
       logarithm refuses to forecast a series with a value that is not positive
   * - ``difference``
     - ``None`` (the season)
     - ``0`` levels, ``1`` first difference, or the ``season`` (seasonal difference); any other lag
       is refused
   * - ``calendar``
     - ``True``
     - add the sine and cosine of the position in the season as inputs
   * - ``standardize``
     - ``True``
     - centre and scale the differenced series on the training data
   * - ``seeds``
     - ``(0,)``
     - one network per seed (at most 100); their forecasts are averaged
   * - ``description``
     - ``""``
     - free text copied into the module's docstring (escaped: it cannot become code)

Which networks it takes
-----------------------

The generator checks the model before writing anything, and lists every problem at once. The
first layer (after any dropout) must read the window, and the last must be a ``LinearLayer``
without an activation that returns ``horizon`` values. With the calendar features on, the first
layer is one of:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - First layer
     - Input shape
     - Required size
   * - ``LinearLayer``
     - (batch, features)
     - ``in_features = window + 2``
   * - ``LSTMLayer``, ``GRULayer``, ``SimpleRNNLayer`` (``batch_first``)
     - (batch, window, 3)
     - ``input_size = 3``
   * - ``Conv1D``
     - (batch, 3, window)
     - ``in_channels = 3``

Without calendar features, ``2`` becomes ``0`` and ``3`` becomes ``1``. Between the first and the
last layer, any layer or tensor operation the PyTorch generator supports may be used (stacked or
bidirectional recurrent layers, pooling, flatten, concatenation of branches, permutes...).
Classifiers (an embedding or a 2D convolution first, a softmax last, the ``crossentropy`` loss)
are refused with the reasons.

The generated module
--------------------

``forecaster_<model name>.py`` holds the network class written by the PyTorch templates and:

- **windowing**: every origin with a full window is a training example; targets not observed yet
  (the last origins) are masked out of the loss, so the end of the series still trains the network;
- **transforms** and their inverse: logarithm, first or seasonal difference, standardisation;
- **calendar features**: the sine and cosine of the season position;
- **deterministic training**: a seed per network and a seeded shuffle, without touching the global
  random state. The same model, configuration and series give the same forecast;
- ``fit``, ``Forecaster.predict`` (any number of steps: beyond ``horizon`` the forecasts feed the
  next window), ``save`` and ``load`` (weights only, no timestamps; ``load`` refuses a file saved by
  another model);
- ``backtest`` (rolling origin: each forecast is trained only on the values up to its origin),
  ``seasonal_naive`` and ``mase`` (an origin where the seasonal naive has no in-sample error leaves
  the MASE undefined: it is left out, with a warning);
- a command line (``main``).

Generation is deterministic: the same model and configuration give the same file, byte for byte.

.. note::

   A small network trained on a single series of a few hundred values often does no better than
   the seasonal naive forecast: compare it with ``backtest`` and ``mase`` before relying on it. In
   one measurement (October 2026: Luxembourg's monthly exports of goods, January 1999 to June 2026,
   60 epochs, one seed, 12 rolling origins), the eight networks tried (two MLPs, two GRUs, an
   LSTM, a simple RNN and two convolutional networks) all beat the seasonal naive forecast one
   month ahead, with a MASE of 0.36 to 0.52, but none did on average over the 12 months: 0.81 to
   1.16, against 0.72 for the seasonal naive. ``fit([...])`` trains one model on several related series; that was not
   measured.

Web Modeling Editor
-------------------

In an NN diagram, **Generate → PyTorch → Time-Series Forecaster** opens a settings form: the
window, the horizon, the season, the seed, the differencing (levels, first difference or seasonal),
the logarithm, the season features, standardisation and ``channel_last``, pre-filled with the
defaults below. The window has to be given: it cannot be read from a recurrent or convolutional
first layer. When generation fails (for example a horizon that does not match the last layer), the
form stays open with the backend's message. The editor sends the form as the generation ``config``
of the ``forecaster`` generator and downloads ``forecaster_<model name>.py``.

The backend reads the ``config`` with ``ForecastConfig.from_dict``:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Default
   * - ``window``
     - integer, 1 to 2147483647
     - 24
   * - ``horizon``
     - integer, 1 to 2147483647 (the size of the last layer)
     - 12
   * - ``season``
     - integer, 1 to 2147483647
     - 12
   * - ``difference``
     - integer: 0, 1 or the season; or null
     - null (the season)
   * - ``log``, ``calendar``, ``standardize``
     - true or false
     - true
   * - ``seed`` (one) or ``seeds`` (a list of at most 100; not both)
     - integers
     - 0
   * - ``description``
     - text, escaped into the module's docstring
     - empty
   * - ``channel_last``
     - true or false (as in :doc:`pytorch`)
     - false

An unknown key, a value of the wrong type, or a network that does not fit the settings (its first
layer must read the window, its last return the horizon) is answered with a 400 that names every
problem.

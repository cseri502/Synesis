using System.Numerics.Tensors;
using Synesis.Core.Activations;
using Synesis.Core.Initializers;
using Synesis.Core.Layers;
using Synesis.Core.Losses;
using Synesis.Core.Network;
using Synesis.Core.Optimizers;

var random = new Random(42);

float[] xorInputData =
[
    0f, 0f,
    0f, 1f,
    1f, 0f,
    1f, 1f,
];

float[] xorTargetData =
[
    0f,
    1f,
    1f,
    0f,
];

var xorInputs  = Tensor.Create<float>(xorInputData,  [(nint)4, (nint)2]);
var xorTargets = Tensor.Create<float>(xorTargetData, [(nint)4, (nint)1]);

var net = new NeuralNetwork(
    new MeanSquaredError(), new SGD(learningRate: 0.5f, momentum: 0.9f))
    .AddLayer(new DenseLayer(inputSize: 2, outputSize: 4, new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new Tanh()))
    .AddLayer(new DenseLayer(inputSize: 4, outputSize: 1,  new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new Sigmoid()));
    
var history = net.Train(xorInputs, xorTargets, epochs: 2000, batchSize: 4, random: random);

Console.WriteLine($"Loss: {history.Records[^1].Loss:F6}");

var predictions = net.Predict(xorInputs);
var predictedFlat = new float[4];
predictions.FlattenTo(predictedFlat);

for (int i = 0; i < 4; i++)
    Console.WriteLine($"Bemenet: [{xorInputData[i * 2]}, {xorInputData[i * 2 + 1]}] -> Predikció: {predictedFlat[i]:F4}");
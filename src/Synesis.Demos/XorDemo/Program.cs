using System.Numerics.Tensors;
using Synesis.Core.Activations;
using Synesis.Core.Initializers;
using Synesis.Core.Layers;
using Synesis.Core.Losses;
using Synesis.Core.Network;
using Synesis.Core.Optimizers;
using Synesis.Core.Serialization;

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

const string modelPath = "xor-model.json";

var net = new NeuralNetwork(
        new MeanSquaredError(), new Adam(learningRate: 0.01f))
    .AddLayer(new DenseLayer(inputSize: 2, outputSize: 4, new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new SiLU()))
    .AddLayer(new DenseLayer(inputSize: 4, outputSize: 1, new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new Sigmoid()));

const int logEvery = 100;
net.EpochCompleted += (epoch, loss) =>
{
    if (epoch % logEvery == 0)
        Console.WriteLine($"Epoch {epoch,5} | Loss: {loss:F6}");
};

if (File.Exists(modelPath))
{
    Console.WriteLine($"Betöltés modellből: {modelPath}");
    ModelSerializer.Load(net, modelPath);
}
else
{
    var history = net.Train(xorInputs, xorTargets, epochs: 2000, batchSize: 4, random: random);
    Console.WriteLine($"Loss: {history.Records[^1].Loss:F6}");

    ModelSerializer.Save(net, modelPath);
    Console.WriteLine($"Modell elmentve: {modelPath}");
}

var predictions = net.Predict(xorInputs);
var predictedFlat = new float[4];
predictions.FlattenTo(predictedFlat);

for (int i = 0; i < 4; i++)
    Console.WriteLine($"Bemenet: [{xorInputData[i * 2]}, {xorInputData[i * 2 + 1]}] -> Predikció: {predictedFlat[i]:F4}");


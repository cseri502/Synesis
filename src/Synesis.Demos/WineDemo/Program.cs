using System.Numerics.Tensors;
using Synesis.Core.Activations;
using Synesis.Core.Data;
using Synesis.Core.Helpers;
using Synesis.Core.Initializers;
using Synesis.Core.Layers;
using Synesis.Core.Losses;
using Synesis.Core.Network;
using Synesis.Core.Optimizers;
using Synesis.Core.Serialization;
using WineDemo;

var random = new Random(42);

const string dataPath = "data/wine.csv";
var (rawInputs, targets) = WineDatasetLoader.Load(dataPath);

int sampleCount = (int)rawInputs.Lengths[0];
int featureCount = (int)rawInputs.Lengths[1];

var inputFlat = new float[sampleCount * featureCount];
rawInputs.FlattenTo(inputFlat);
FeatureScaler.StandardizeInPlace(inputFlat, sampleCount, featureCount);
var inputs = Tensor.Create<float>(inputFlat, [(nint)sampleCount, (nint)featureCount]);

var (trainInputs, trainTargets, testInputs, testTargets) =
    DataSplitter.Split(inputs, targets, testFraction: 0.25f, random);

Console.WriteLine($"Train minták: {trainInputs.Lengths[0]}, Test minták: {testInputs.Lengths[0]}");

var net = new NeuralNetwork(
        new SoftmaxCrossEntropyLoss(),
        new AdamW(learningRate: 0.001f, weightDecay: 0.01f))
    .AddLayer(new DenseLayer(inputSize: 13, outputSize: 16, new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new SiLU()))
    .AddLayer(new DenseLayer(inputSize: 16, outputSize: 8, new XavierInitializer(), random))
    .AddLayer(new ActivationLayer(new SiLU()))
    .AddLayer(new DropoutLayer(dropRate: 0.2f, random))
    .AddLayer(new DenseLayer(inputSize: 8, outputSize: 3, new XavierInitializer(), random));

const int logEvery = 50;
net.EpochCompleted += (epoch, loss) =>
{
    if (epoch % logEvery == 0)
        Console.WriteLine($"Epoch {epoch,5} | Loss: {loss:F6}");
};

var history = net.Train(trainInputs, trainTargets, epochs: 200, batchSize: 16, random: random);
Console.WriteLine($"Végső train loss: {history.Records[^1].Loss:F6}");

var testPredictions = net.Predict(testInputs);
float testAccuracy = Metrics.Accuracy(testPredictions, testTargets);
Console.WriteLine($"Teszt pontosság: {testAccuracy:P2}");

const string modelPath = "wine-model.json";
ModelSerializer.Save(net, modelPath);
Console.WriteLine($"Modell elmentve: {modelPath}");
using System.Numerics.Tensors;
using Synesis.Core.Activations;
using Synesis.Core.Layers;
using Synesis.Core.Losses;
using Synesis.Core.Optimizers;

namespace Synesis.Core.Network;

public sealed class NeuralNetwork
{
    private readonly List<ILayer> _layers = [];
    private readonly ILossFunction _loss;
    private readonly IOptimizer _optimizer;
    private readonly IActivationFunction? _predictionActivation;
    
    public event Action<int, float>? EpochCompleted;
    public IReadOnlyList<ILayer> Layers => _layers;

    public NeuralNetwork(ILossFunction loss, IOptimizer optimizer, IActivationFunction? predictionActivation = null)
    {
        _loss = loss;
        _optimizer = optimizer;
        _predictionActivation = predictionActivation;
    }

    public NeuralNetwork AddLayer(ILayer layer)
    {
        _layers.Add(layer);
        return this;
    }

    public Tensor<float> Predict(Tensor<float> input)
    {
        var current = input;

        foreach (var layer in _layers)
            current = layer.Forward(current, isTraining: false);
        
        return _predictionActivation is null ? current : _predictionActivation.Forward(current);
    }

    public TrainingHistory Train(Tensor<float> inputs, Tensor<float> targets, int epochs, int batchSize,
        Random? random = null, bool shuffle = true)
    {
        random ??= Random.Shared;
        var history = new TrainingHistory();
        
        int sampleCount = (int)inputs.Lengths[0];
        int inputFeatures = (int)inputs.Lengths[1];
        int targetFeatures = (int)targets.Lengths[1];
        
        var inputFlat = new float[sampleCount * inputFeatures];
        inputs.FlattenTo(inputFlat);
        var targetFlat = new float[sampleCount * targetFeatures];
        targets.FlattenTo(targetFlat);
        
        var indices = Enumerable.Range(0, sampleCount).ToArray();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (shuffle)
                Shuffle(indices, random);
            
            float epochLossSum = 0f;
            int batchCount = 0;

            for (int start = 0; start < sampleCount; start++)
            {
                int currentBatchSize = Math.Min(batchSize, sampleCount - start);
                
                var batchInput = GatherBatch(inputFlat, indices, start, currentBatchSize, inputFeatures);
                var batchTarget = GatherBatch(targetFlat, indices, start, currentBatchSize, targetFeatures);
                
                var inputTensor = Tensor.Create<float>(batchInput, [(nint)currentBatchSize, (nint)inputFeatures]);
                var targetTensor = Tensor.Create<float>(batchTarget, [(nint)currentBatchSize, (nint)targetFeatures]);
                
                var predictions = inputTensor;
                foreach (var layer in _layers)
                    predictions = layer.Forward(predictions, isTraining: true);
                
                epochLossSum += _loss.Compute(predictions, targetTensor);
                batchCount++;
                
                var grad = _loss.Gradient(predictions, targetTensor);
                for (int i = _layers.Count - 1; i >= 0; i--)
                    grad = _layers[i].Backward(grad);
                
                foreach (var layer in _layers)
                    layer.UpdateParameters(_optimizer);
            }
            
            float avgLoss = epochLossSum / batchCount;
            history.RecordEpoch(epoch, avgLoss);
            
            EpochCompleted?.Invoke(epoch, avgLoss);
        }
        
        return history;
    }
    
    private static float[] GatherBatch(float[] source, int[] indices, int start, int count, int featureSize)
    {
        var batch = new float[count * featureSize];
        
        for (int i = 0; i < count; i++)
        {
            int sourceIndex = indices[start + i];
            Array.Copy(source, sourceIndex * featureSize, batch, i * featureSize, featureSize);
        }
        
        return batch;
    }
    
    // Fisher–Yates
    private static void Shuffle(int[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
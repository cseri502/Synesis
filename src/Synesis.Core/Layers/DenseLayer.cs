using System.Numerics.Tensors;
using Synesis.Core.Initializers;
using Synesis.Core.Optimizers;

namespace Synesis.Core.Layers;

public sealed class DenseLayer : ILayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    
    private readonly float[] _weights;
    private readonly float[] _bias;
    
    private float[] _lastInput = [];
    private int _lastBatchSize;
    
    private readonly float[] _weightGradients;
    private readonly float[] _biasGradients;

    public DenseLayer(int inputSize, int outputSize, IWeightInitializer initializer)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        
        _weights = new float[outputSize * inputSize];
        _bias = new float[outputSize];
        _weightGradients = new float[_weights.Length];
        _biasGradients = new float[_bias.Length];
        
        initializer.Initialize(_weights, fanIn: inputSize, fanOut: outputSize);
    }

    public Tensor<float> Forward(Tensor<float> input, bool isTraining)
    {
        throw new NotImplementedException();
    }

    public Tensor<float> Backward(Tensor<float> gradOutput)
    {
        throw new NotImplementedException();
    }

    public void UpdateParameters(IOptimizer optimizer)
    {
        throw new NotImplementedException();
    }
}
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
        var batchSize = (int)input.Lengths[0];
        _lastBatchSize = batchSize;
        
        _lastInput = new  float[batchSize * _inputSize];
        input.FlattenTo(_lastInput);
        
        var output = new float[batchSize * _outputSize];

        for (int b = 0; b < batchSize; b++)
        {
            var inputRow = _lastInput.AsSpan(b * _inputSize, _inputSize);

            for (int o = 0; o < _outputSize; o++)
            {
                var weightRow = _weights.AsSpan(o * _inputSize, _inputSize);
                output[b * _outputSize + o] = TensorPrimitives.Dot(inputRow, weightRow) + _bias[o];
            }
        }
        
        return Tensor.Create(output, [batchSize, _outputSize]);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput)
    {
        var gradOutFlat = new float[_lastBatchSize *  _outputSize];
        gradOutput.FlattenTo(gradOutFlat);
        
        Array.Clear(_weightGradients);
        Array.Clear(_biasGradients);
        
        var gradInput = new float[_lastBatchSize * _inputSize];

        for (int b = 0; b < _lastBatchSize; b++)
        {
            var inputRow = _lastInput.AsSpan(b * _inputSize, _inputSize);
            var gradInputRow = gradInput.AsSpan(b * _inputSize, _inputSize);
            
            for (int o = 0; o < _outputSize; o++)
            {
                float grad = gradOutFlat[b * _outputSize + o];
                var weightRow = _weights.AsSpan(o * _inputSize, _inputSize);
                var weightGradRow = _weightGradients.AsSpan(o * _inputSize, _inputSize);

                // dL/dW[o,i] = grad * input[b,i]   (súly-gradiens)
                // dL/dInput[b,i] += grad * W[o,i]   (bemeneti gradiens, lánc-szabály)
                for (int i = 0; i < _inputSize; i++)
                {
                    weightGradRow[i] += grad * inputRow[i];
                    gradInputRow[i]  += grad * weightRow[i];
                }

                _biasGradients[o] += grad;
            }
        }
        
        return Tensor.Create<float>(gradInput, [_lastBatchSize, _inputSize]);
    }

    public void UpdateParameters(IOptimizer optimizer)
    {
        optimizer.Update(_weights, _weightGradients);
        optimizer.Update(_bias, _biasGradients);
    }
}
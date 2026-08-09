using System.Numerics.Tensors;
using Synesis.Core.Helpers;
using Synesis.Core.Optimizers;

namespace Synesis.Core.Layers;

public sealed class DropoutLayer : ILayer
{
    private readonly float _dropRate;
    private readonly Random _random;
    
    private float[] _lastMask = [];

    public DropoutLayer(float dropRate, Random? random = null)
    {
        if (dropRate is < 0f or >= 1f)
            throw new ArgumentOutOfRangeException(nameof(dropRate), "Drop rate must be specified between 0 and 1.");

        _dropRate = dropRate;
        _random = random ?? Random.Shared;
    }
    
    public Tensor<float> Forward(Tensor<float> input, bool isTraining)
    {
        if (!isTraining || _dropRate == 0f)
        {
            _lastMask = Array.Empty<float>();
            return input;
        }

        int elementCount = TensorShapeHelpers.ElementCount(input.Lengths);

        var flatInput = new float[elementCount];
        input.FlattenTo(flatInput);

        float keepProbability = 1f - _dropRate;
        var mask = new float[elementCount];
        var output = new float[elementCount];

        for (int i = 0; i < elementCount; i++)
        {
            bool keep = _random.NextDouble() < keepProbability;
            mask[i] = keep ? 1f / keepProbability : 0f;
            output[i] = flatInput[i] * mask[i];
        }

        _lastMask = mask;
        return Tensor.Create<float>(output, input.Lengths);
    }
    
    public Tensor<float> Backward(Tensor<float> gradOutput)
    {
        if (_lastMask.Length == 0)
            return gradOutput;

        var flatGrad = new float[_lastMask.Length];
        gradOutput.FlattenTo(flatGrad);

        var gradInput = new float[_lastMask.Length];
        for (int i = 0; i < _lastMask.Length; i++)
            gradInput[i] = flatGrad[i] * _lastMask[i];

        return Tensor.Create<float>(gradInput, gradOutput.Lengths);
    }

    public void UpdateParameters(IOptimizer optimizer) {}
}
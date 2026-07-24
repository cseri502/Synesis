using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class ReLU : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input)
        => Tensor.Max<float>(input.AsReadOnlyTensorSpan(), 0f);

    public Tensor<float> Derivative(Tensor<float> input)
    {
        var flat = new float[input.Lengths.ToArray().Aggregate(1, (a, b) => a * (int)b)];
        input.FlattenTo(flat);

        for (var i = 0; i < flat.Length; i++)
            flat[i] = flat[i] > 0f ? 1f : 0f;

        return Tensor.Create(flat, input.Lengths);
    }
}
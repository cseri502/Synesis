using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class Tanh : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input)
        => Tensor.Tanh<float>(input.AsReadOnlyTensorSpan());

    public Tensor<float> Derivative(Tensor<float> input)
    {
        // f'(x) = 1 - tanh(x)^2
        var t = Forward(input);
        var tSquared = Tensor.Multiply<float>(t.AsReadOnlyTensorSpan(), t.AsReadOnlyTensorSpan());
        return Tensor.Subtract<float>(1f, tSquared.AsReadOnlyTensorSpan());
    }
}
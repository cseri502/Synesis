using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class Sigmoid : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input) => Tensor.Sigmoid(input.AsReadOnlyTensorSpan());

    public Tensor<float> Derivative(Tensor<float> input)
    {
        var s = Tensor.Sigmoid(input.AsReadOnlyTensorSpan());
        var one = Tensor.Create<float>([1f], []);
        return Tensor.Multiply<float>(s, Tensor.Subtract<float>(one, s));
    }
}
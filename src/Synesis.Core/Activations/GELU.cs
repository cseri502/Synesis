using System.Numerics.Tensors;

namespace Synesis.Core.Activations;

public sealed class GELU : IActivationFunction
{
    public Tensor<float> Forward(Tensor<float> input)
    {
        var flat = new float[input.Lengths.ToArray().Aggregate(1, (a, b) => a * (int)b)];
        input.FlattenTo(flat);

        const float sqrtTwoOverPi = 0.7978845608f; // Math.Sqrt(2.0 / Math.PI)
        const float c = 0.044715f;

        for (var i = 0; i < flat.Length; i++)
        {
            var x = flat[i];
            var inner = sqrtTwoOverPi * (x + c * x * x * x);
            flat[i] = 0.5f * x * (1f + (float)Math.Tanh(inner));
        }

        return Tensor.Create(flat, input.Lengths);
    }

    public Tensor<float> Derivative(Tensor<float> input)
    {
        var flat = new float[input.Lengths.ToArray().Aggregate(1, (a, b) => a * (int)b)];
        input.FlattenTo(flat);

        const float sqrtTwoOverPi = 0.7978845608f;
        const float c = 0.044715f;

        for (var i = 0; i < flat.Length; i++)
        {
            var x = flat[i];
            var x3 = x * x * x;
            var inner = sqrtTwoOverPi * (x + c * x3);
            var tanhInner = (float)Math.Tanh(inner);
            var sech2 = 1f - tanhInner * tanhInner; // 1 - tanh^2(x)

            var dInner = sqrtTwoOverPi * (1f + 3f * c * x * x);
            
            // f'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * dInner
            flat[i] = 0.5f * (1f + tanhInner) + 0.5f * x * sech2 * dInner;
        }

        return Tensor.Create(flat, input.Lengths);
    }
}
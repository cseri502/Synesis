using System.Numerics.Tensors;

namespace Synesis.Core.Losses;

public interface ILossFunction
{
    float Compute(Tensor<float> predictions, Tensor<float> targets);
    Tensor<float> Gradient(Tensor<float> predictions, Tensor<float> targets);
}
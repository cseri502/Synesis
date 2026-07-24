using System.Numerics.Tensors;
using Synesis.Core.Optimizers;

namespace Synesis.Core.Layers;

public interface ILayer
{
    Tensor<float> Forward(Tensor<float> input, bool isTraining);
    Tensor<float> Backward(Tensor<float> gradOutput);
    void UpdateParameters(IOptimizer optimizer);
}
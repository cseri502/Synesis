namespace Synesis.Core.Optimizers;

public interface IOptimizer
{
    void Update(float[] parameters, float[] gradients);
}
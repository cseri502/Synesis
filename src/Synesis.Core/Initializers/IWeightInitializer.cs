namespace Synesis.Core.Initializers;

public interface IWeightInitializer
{
    void Initialize(float[] weights, int fanIn, int fanOut);
}
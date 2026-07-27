namespace Synesis.Core.Initializers;

public interface IWeightInitializer
{
    void Initialize(Span<float> weights, int fanIn, int fanOut, Random random);
}
namespace Synesis.Core.Initializers;

public sealed class XavierInitializer : IWeightInitializer
{
    public void Initialize(Span<float> weights, int fanIn, int fanOut, Random random)
    {
        float x = MathF.Sqrt(6f / (fanIn + fanOut));

        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(random.NextDouble() * 2.0 - 1) * x;
    }
}
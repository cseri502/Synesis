namespace Synesis.Core.Initializers;

public sealed class HeInitializer : IWeightInitializer
{
    public void Initialize(Span<float> weights, int fanIn, int fanOut, Random random)
    {
        float std = MathF.Sqrt(2f / fanIn);

        for (int i = 0; i < weights.Length; i++)
            weights[i] = SampleStandardGaussian(random) * std;
    }
    
    private static float SampleStandardGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(); // (0,1]
        double u2 = random.NextDouble();
        
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
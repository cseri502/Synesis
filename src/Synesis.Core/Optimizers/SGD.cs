using System.Runtime.CompilerServices;

namespace Synesis.Core.Optimizers;

public sealed class SGD : IOptimizer
{
    private readonly float _learningRate;
    private readonly float _momentum;
    
    private readonly ConditionalWeakTable<float[], float[]> _velocities = new();

    public SGD(float learningRate = 0.01f, float momentum = 0.9f)
    {
        _learningRate = learningRate;
        _momentum = momentum;
    } 

    public void Update(float[] parameters, float[] gradients)
    {
        float[] velocity = _velocities.GetValue(parameters, key => new float[key.Length]);
        
        for (int i = 0; i < parameters.Length; i++)
        {
            velocity[i] = _momentum * velocity[i] + gradients[i];
            parameters[i] -= _learningRate * velocity[i];
        }
    }
}
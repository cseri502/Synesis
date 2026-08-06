using System.Runtime.CompilerServices;

namespace Synesis.Core.Optimizers;

public sealed class AdamW : IOptimizer
{
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    
    private sealed class State(int size)
    {
        public readonly float[] FirstMoment = new float[size];
        public readonly float[] SecondMoment = new float[size];
        public int Timestep;
    }

    private readonly ConditionalWeakTable<float[], State> _states = new();
    
    public AdamW(
        float learningRate = 0.001f, 
        float beta1 = 0.9f, 
        float beta2 = 0.999f, 
        float epsilon = 1e-8f, 
        float weightDecay = 0.01f)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }
    
    public void Update(float[] parameters, float[] gradients)
    {
        var state = _states.GetValue(parameters, key => new State(key.Length));
        state.Timestep++;

        float biasCorrection1 = 1f - MathF.Pow(_beta1, state.Timestep);
        float biasCorrection2 = 1f - MathF.Pow(_beta2, state.Timestep);

        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] -= _learningRate * _weightDecay * parameters[i];

            state.FirstMoment[i] = _beta1 * state.FirstMoment[i] + (1f - _beta1) * gradients[i];
            state.SecondMoment[i] = _beta2 * state.SecondMoment[i] + (1f - _beta2) * gradients[i] * gradients[i];

            float mHat = state.FirstMoment[i] / biasCorrection1;
            float vHat = state.SecondMoment[i] / biasCorrection2;

            parameters[i] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }
}
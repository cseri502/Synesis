using System.Runtime.CompilerServices;

namespace Synesis.Core.Optimizers;

public sealed class Adam : IOptimizer
{
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    
    private sealed class State
    {
        public readonly float[] FirstMoment;
        public readonly float[] SecondMoment;
        public int Timestep;
    
        public State(int size)
        {
            FirstMoment = new float[size];
            SecondMoment = new float[size];
        }
    }
    
    private readonly ConditionalWeakTable<float[], State> _states = new();
    
    public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    public void Update(float[] parameters, float[] gradients)
    {
        var state = _states.GetValue(parameters, key => new State(key.Length));
        state.Timestep++;

        float biasCorrection1 = 1f - MathF.Pow(_beta1, state.Timestep);
        float biasCorrection2 = 1f - MathF.Pow(_beta2, state.Timestep);

        for (int i = 0; i < parameters.Length; i++)
        {
            state.FirstMoment[i] = _beta1 * state.FirstMoment[i] + (1f - _beta1) * gradients[i];
            state.SecondMoment[i] = _beta2 * state.SecondMoment[i] + (1f - _beta2) * gradients[i] * gradients[i];
            
            float mHat = state.FirstMoment[i] /  biasCorrection1;
            float vHat = state.SecondMoment[i] /  biasCorrection2;
            
            parameters[i] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }
}
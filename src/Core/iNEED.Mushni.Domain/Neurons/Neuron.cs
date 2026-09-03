namespace iNEED.Mushni.Domain.Neurons;

public class Neuron
{
    private double[] _weights;
    private double _bias;
    public Neuron(double[] weights, double bias)
    {
        _weights = weights;
        _bias = bias;
    }

    public double Activate(double[] inputs)
    {
        if (inputs.Length != _weights.Length)
            throw new ArgumentException("Input length must match weights length.");

        double weightedSum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            weightedSum += inputs[i] * _weights[i];
        }
        weightedSum += _bias;

        return Sigmoid(weightedSum);
    }

    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public void Train(double[] inputs, double targetOutput, double learningRate)
    {
        double output = Activate(inputs);
        double error = targetOutput - output;

        // Update weights and bias
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] += learningRate * error * inputs[i];
        }
        _bias += learningRate * error;
    }
}
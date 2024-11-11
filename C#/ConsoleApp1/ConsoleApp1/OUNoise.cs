namespace ConsoleApp1;
using Numpy;

public class OUNoise(int size, float mu = 0f, float sigma = 0.1f, float theta = 0.15f)
{
    public Numpy.NDarray Mu { get; private set; } = mu * np.ones(size);
    public Numpy.NDarray State { get; private set; } = np.ones(size);
    public float Sigma { get; private set; } = sigma;
    public float Theta { get; private set; } = theta;
    public int Size { get; private set; } = size;

    public void Reset()
    {
        State = Mu.copy();
    }

    public Numpy.NDarray Sample()
    {
        // dx = (self.theta * (self.mu - self.state)) + (self.sigma * np.random.randn(self.size))
        var dx = (Theta * (Mu - State)) + (Sigma * np.random.randn(Size));
        State += dx;
        return State.copy();
    }
}
namespace ConsoleApp1;
using TorchSharp;

public class Actor(int stateDim, int actionDim, int hiddenDim = 64)
{
    public torch.nn.Module Network { get; private set; } = torch.nn.Sequential(
        torch.nn.Linear(stateDim, actionDim),
        torch.nn.ReLU(),
        torch.nn.Linear(hiddenDim, hiddenDim),
        torch.nn.ReLU(),
        torch.nn.Linear(hiddenDim, actionDim),
        torch.nn.Tanh());

    public torch.nn.Module Forward(Action<torch.nn.Module> x)
    {
        return this.Network.apply(x);
    }
}
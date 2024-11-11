namespace ConsoleApp1;
using TorchSharp;

public class Critic(int stateDim, int actionDim, int hiddenDim = 64)
{
    public torch.nn.Module Network { get; private set; } = torch.nn.Sequential(
        torch.nn.Linear(stateDim + actionDim, actionDim),
        torch.nn.ReLU(),
        torch.nn.Linear(hiddenDim, hiddenDim),
        torch.nn.ReLU(),
        torch.nn.Linear(hiddenDim, 1),
        torch.nn.Tanh());

    public torch.nn.Module Forward(Action<torch.nn.Module> x)
    {
        return Network.apply(x);
    }
}
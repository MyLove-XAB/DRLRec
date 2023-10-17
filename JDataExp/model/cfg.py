import argparse


def get_args():  # config hyper parameters
    parser = argparse.ArgumentParser(description="hyper parameters")

    # PPO parameters
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--ddpgtrain_eps', default=400, type=int, help="episodes of training")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=4, type=int, help='update number')
    parser.add_argument('--actor_lr', default=0.00001, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.001, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.1, type=float, help='policy clip')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--critic_hidden_dim', default=256, type=int, help='critic hidden dim')
    parser.add_argument("--dynamic_hidden", default=64, type=int, help="dynamic info hidden dim")
    parser.add_argument('--ddpgbatch_size', default=128, type=int, help='batch size')

    parser.add_argument('--static_dim', default=88, type=int, help="input features dim, equals to static features dim")
    parser.add_argument('--hidden_dim', default=128, type=int, help='middle hidden dim')
    parser.add_argument("--out_dim", default=64, type=int, help="output dim")

    # Env parameters
    parser.add_argument("--rec_runs", default=10, type=int, help="recommendation rus for each customer")
    parser.add_argument("--env_batch", default=256, type=int, help="the number of users for each time step")
    parser.add_argument("--rec_item_num", default=10, type=int, help="the number of items recommended each step")
    parser.add_argument("--dynamic_dim", default=25, type=int, help="the dimension of dynamic features")

    # System parameters
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")

    args = parser.parse_args()
    return args
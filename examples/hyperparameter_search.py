import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blackjack_lib.agents.Q_agent import QAgent
import pandas as pd
from simulate_Q import evaluate_win_rate, evaluate_Q, plot_training_evaluation_performance

def hyperparameter_search(discounts=[0.9, 0.95, 0.99], 
                         epsilons=[0.1, 0.2, 0.4, 0.6],
                         lr_bases=[5.0, 10.0, 20.0],
                         num_train=50000,
                         num_eval=10000,
                         train_eval_interval=1000,
                         verbose=True):
    
    results = []
    
    total_combinations = len(discounts) * len(epsilons) * len(lr_bases)
    current = 0
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH")
    print(f"{'='*80}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"Discounts: {discounts}")
    print(f"Epsilons: {epsilons}")
    print(f"Learning Rate Bases: {lr_bases}")
    print(f"Training Games: {num_train:,}")
    print(f"Evaluation Games: {num_eval:,}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for discount in discounts:
        for epsilon in epsilons:
            for lr_base in lr_bases:
                current += 1
                
                if verbose:
                    print(f"[{current}/{total_combinations}] Testing: γ={discount:.2f}, ε={epsilon:.2f}, LR_base={lr_base:.1f}...", end=" ")
                
                combo_start = time.time()
                agent = QAgent(discount=discount, lr_base=lr_base)
                agent.Q_run(num_simulation=num_train, epsilon=epsilon, 
                           track_performance=False, eval_interval=train_eval_interval)
                
                win_rate = evaluate_win_rate(agent, num_games=num_eval)
                combo_time = time.time() - combo_start
                
                results.append({
                    'Discount': discount,
                    'Epsilon': epsilon,
                    'LR_Base': lr_base,
                    'Win_Rate': win_rate
                })
                
                if verbose:
                    print(f"Win Rate: {win_rate:.4f} (Time: {combo_time:.2f}s)")
    
    total_time = time.time() - start_time
    
    df = pd.DataFrame(results)
    df = df.sort_values('Win_Rate', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    best_params = df.iloc[0]
    print(f"Best Hyperparameters:")
    print(f"  Discount (γ): {best_params['Discount']:.2f}")
    print(f"  Epsilon (ε): {best_params['Epsilon']:.2f}")
    print(f"  Learning Rate Base: {best_params['LR_Base']:.1f}")
    print(f"  Win Rate: {best_params['Win_Rate']:.4f} ({best_params['Win_Rate']*100:.2f}%)")
    print(f"\nTotal Search Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average Time per Combination: {total_time/total_combinations:.2f}s")
    print(f"{'='*80}\n")
    
    return df, best_params

def plot_best_hyperparameters(best_params, num_train=50000, num_eval=10000, 
                              train_eval_interval=1000, save_path=None):
    print(f"Training and plotting with best hyperparameters...")
    
    train_start = time.time()
    agent = QAgent(discount=best_params['Discount'], lr_base=best_params['LR_Base'])
    agent.Q_run(num_simulation=num_train, epsilon=best_params['Epsilon'], 
               track_performance=True, eval_interval=train_eval_interval)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    eval_start = time.time()
    eval_history = evaluate_Q(agent, num_games=num_eval, track_performance=True)
    eval_time = time.time() - eval_start
    print(f"Evaluation completed in {eval_time:.2f}s")
    print(f"Total time: {train_time + eval_time:.2f}s ({(train_time + eval_time)/60:.2f} minutes)\n")
    
    if save_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(project_root, 'figures', 'q_agent_best_hyperparams.png')
    
    plot_training_evaluation_performance(agent, eval_history, num_train, num_eval,
                                        best_params['Epsilon'], best_params['Discount'], 
                                        train_eval_interval, save_path, 
                                        lr_base=best_params['LR_Base'])

if __name__ == "__main__":
    df_results, best = hyperparameter_search(
        discounts=[0.9, 0.95, 0.99],
        epsilons=[0.1, 0.2, 0.4, 0.6],
        lr_bases=[5.0, 10.0, 20.0],
        num_train=50000,
        num_eval=10000,
        train_eval_interval=1000
    )
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'figures', 'q_agent_best_hyperparams.png')
    
    plot_best_hyperparameters(best, 
                             num_train=50000,
                             num_eval=10000,
                             train_eval_interval=1000,
                             save_path=save_path)


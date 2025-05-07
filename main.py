import torch
import torch.nn as nn
import torch.optim as optim
import random
import tkinter as tk

#############################################
# Global Vocabulary and Text Processing
#############################################

# Vocabulary covers key procedural words and digits.
vocab = [
           "multiply", "add", "carry", "and", "combine", "do", "nothing", "subtract",
           "output", "ones", "digit", "tens", "hundreds"
       ] + [str(i) for i in range(10)]
vocab = [w.lower() for w in vocab]
vocab_size = len(vocab)
vocab_dict = {word: idx for idx, word in enumerate(vocab)}


def preprocess(text):
   text = text.lower()
   for symbol in [":", "*", "+", "-", "="]:
       text = text.replace(symbol, f" {symbol} ")
   return text.split()

def encode_text(text):
    tokens = preprocess(text)
    vec = torch.zeros(vocab_size)
    for token in tokens:
        if token in vocab_dict:
            vec[vocab_dict[token]] += 1.0
    return vec


def encode_state_text(problem_text, chain_text, step_index=None, carry1=None, carry2=None):
    state_vec = torch.cat([encode_text(problem_text), encode_text(chain_text)])
    if step_index is not None and carry1 is not None and carry2 is not None:
        extra_vec = torch.tensor([
            step_index / 6.0,
            carry1 / 9.0,
            carry2 / 9.0
        ])
        state_vec = torch.cat([state_vec, extra_vec])
    return state_vec



"""
    Encode state by combining:
    - Problem text (bag of words)
    - Chain text (bag of words)
    - Step index (normalized)
    - Carry1 and Carry2 values (normalized)
"""



#############################################
# Transformer Policy Network
#############################################

class TransformerPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, d_model)  # 这里已经是 state + action 之后的维度
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, state_action):
        x = self.input_proj(state_action).unsqueeze(0).unsqueeze(0)
        x = self.transformer(x)
        return self.output_layer(x.squeeze(0).squeeze(0))


#############################################
# RL Agent (Using Transformer)
#############################################

class RLAgentSimple:
    def __init__(self, lr=1e-3, state_dim=69, num_actions=7):
            self.num_actions = num_actions
            self.state_dim = state_dim
            self.input_dim = self.state_dim + self.num_actions  # 拼接后总输入维度
            self.policy_net = TransformerPolicyNetwork(input_dim=self.input_dim, output_dim=num_actions)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.episode_log_probs = []
            self.loss_terms = []

    def choose_action(self, state, env):
        action_scores = []
        for action_index in range(env.num_actions):
            action_tensor = torch.zeros(env.num_actions)
            action_tensor[action_index] = 1.0
            input_tensor = torch.cat([state, action_tensor])
            score = self.policy_net(input_tensor)
            action_scores.append(score)
        scores_tensor = torch.stack(action_scores).view(-1)
        probs = torch.softmax(scores_tensor, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.episode_log_probs.append(dist.log_prob(action))
        return action.item()

    def accumulate_loss(self, reward):
        if self.episode_log_probs:
            # 添加当前 step 的负 log 概率 × reward
            self.loss_terms.append(-self.episode_log_probs[-1] * reward)

    def finalize_episode(self, final_reward):
        # 给整条链的奖励加 bonus
        bonus = sum([-lp * final_reward for lp in self.episode_log_probs])
        self.loss_terms.append(bonus)

        # 累加所有损失
        total_loss = torch.stack(self.loss_terms).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 清空记录
        self.episode_log_probs = []
        self.loss_terms = []


 #############################################
        # Multiplication Module for Three-Digit x One-Digit
#############################################

def generate_multiplication_problem_three():
            """
            Generates a multiplication problem:
              - A: three-digit number (100-999)
              - M: one-digit number (1-9)
            A human-like six-step procedure is created as follows:
              1. "multiply <ones> by <M>"
              2. "multiply <tens> by <M> with carry <carry1>"
              3. "multiply <hundreds> by <M> with carry <carry2>"
              4. "output ones digit: <d1>"
              5. "output tens digit: <d2>"
              6. "output hundreds digits: <P3>"
            where the correct intermediate tokens are generated using arithmetic.

            Note: The multiplication teacher will no longer recompute these values;
                  it only compares the agent’s chain (a sequence of text tokens) to the correct chain.
            """
            A = random.randint(100, 999)
            M = random.randint(1, 9)
            correct_answer = A * M
            problem_text = f"mul: multiply {A} x {M}"

            ones = A % 10
            tens = (A // 10) % 10
            hundreds = A // 100

            # Step 1:
            P1 = ones * M
            d1 = P1 % 10
            carry1 = P1 // 10
            step1 = f"multiply {ones} by {M}"

            # Step 2:
            P2 = tens * M + carry1
            d2 = P2 % 10
            carry2 = P2 // 10
            step2 = f"multiply {tens} by {M} with carry {carry1}"

            # Step 3:
            P3 = hundreds * M + carry2
            step3 = f"multiply {hundreds} by {M} with carry {carry2}"

            # Output steps:
            step4 = f"output ones digit: {d1}"
            step5 = f"output tens digit: {d2}"
            step6 = f"output hundreds digits: {P3}"

            # The correct procedure as a chain of strings (all lowercase)
            correct_steps = [step1.lower(), step2.lower(), step3.lower(), step4.lower(), step5.lower(), step6.lower()]
            # Allowed actions: the 6 correct ones, plus one extra dummy option to pad to a total of 7.
            allowed_actions = correct_steps + ["dummy"]
            return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, M, carry1, carry2


class MultiplicationTeacherThree:
            """
            Revised Multiplication Teacher.
            Instead of parsing numbers and performing arithmetic, it only compares
            the agent’s chain of actions (text strings) with the pre-generated correct chain.
            This forces the RL agent to learn the full procedure from the text tokens.
            """

            def __init__(self, correct_steps, correct_answer, A, M):
                self.correct_steps = correct_steps
                self.correct_answer = correct_answer
                self.A = A
                self.M = M

            def evaluate_solution(self, chain):
                reward = 0.0
                feedback_lines = []

                stages = ["Ones multiplication", "Tens multiplication", "Hundreds multiplication",
                          "Output ones digit", "Output tens digit", "Output hundreds digits"]

                for i, correct_step in enumerate(self.correct_steps):
                    stage_name = stages[i] if i < len(stages) else f"Step {i + 1}"

                    if i < len(chain):
                        student_step = chain[i]
                        if student_step == correct_step:
                            reward += 8.0  # 正确步骤奖励
                            feedback_lines.append(f"✔ {stage_name} correct: {student_step}")
                        elif student_step.split()[0] == correct_step.split()[0]:
                            reward += 2.0  # 行动类型正确但内容错误
                            feedback_lines.append(
                                f"❗ {stage_name} action type correct but wrong details.\nExpected: {correct_step}\nGot: {student_step}")
                        else:
                            reward -= 4.0
                            feedback_lines.append(
                                f"✘ {stage_name} wrong.\nExpected: {correct_step}\nGot: {student_step}")
                    else:
                        reward -= 2.0
                        feedback_lines.append(f"⚠ Missing {stage_name}. Expected: {correct_step}")

                # 计算顺序奖励
                correct_order = True
                for expected, actual in zip(self.correct_steps, chain):
                    if expected != actual:
                        correct_order = False
                        break
                if correct_order:
                    reward += 10.0
                    feedback_lines.append("🌟 All steps in correct order!")

                # 完全正确奖励
                if chain == self.correct_steps:
                    reward += 20.0
                    feedback_lines.append(f"\n🎉 Full solution correct! Product: {self.correct_answer}")
                else:
                    feedback_lines.append("\n❌ Final result: Procedure not fully correct.")

                return "\n".join(feedback_lines), reward


class MultiplicationEnvThree:
        def __init__(self):
            self.reset()  # 正常调用

        def reset(self):  # ✅ 注意：这个函数不能缩进在 __init__ 里面
            (self.problem_text,
             self.allowed_actions,
             self.correct_steps,
             self.correct_answer,
             self.A,
             self.M,
             self.carry1,
             self.carry2) = generate_multiplication_problem_three()

            self.num_actions = len(self.allowed_actions)
            self.teacher = MultiplicationTeacherThree(self.correct_steps, self.correct_answer, self.A, self.M)
            self.chain = []
            self.chain_text = ""
            self.max_steps = len(self.correct_steps)
            return self.get_state()

        def get_state(self):
                step_index = len(self.chain)
                return encode_state_text(
                    self.problem_text,
                    self.chain_text,
                    step_index=step_index,
                    carry1=self.carry1,
                    carry2=self.carry2
                )

        def step(self, action_index):
                action = self.allowed_actions[action_index]
                self.chain.append(action)
                self.chain_text = (self.chain_text + " " + action) if self.chain_text else action
                done = (len(self.chain) >= self.max_steps)

                reward = 0.0
                current_step = len(self.chain) - 1

                # ✅ 主要奖励逻辑：如果当前步骤正确，加正奖励，否则减分
                if current_step < len(self.correct_steps):
                    if self.chain[current_step] == self.correct_steps[current_step]:
                        reward += 10.0
                    else:
                        reward -= 5.0

                # ✅ 惩罚重复动作（最多允许重复1次）
                if self.chain.count(action) > 2:
                    reward -= 3.0

                # ✅ 若超过最大长度，惩罚并提前结束
                if len(self.chain) > self.max_steps:
                    reward -= 5.0
                    done = True

                return self.get_state(), reward, done, ""


class CompositeMathEnv:
    def __init__(self):
        self.task = "mul"
        self.env = MultiplicationEnvThree()
        self.reset()

    def reset(self):
        self.task = "mul"
        self.env = MultiplicationEnvThree()
        return self.env.get_state()

    def get_state(self):
        return self.env.get_state()

    def step(self, action_index):
        return self.env.step(action_index)
#############################################
# Training Loop (Composite)
#############################################

def train_agent(num_episodes=5000):
   env = CompositeMathEnv()
   state_dim = 2 * vocab_size + 3  # problem + chain + step_index/carry1/carry2
   num_actions = env.env.num_actions
   agent = RLAgentSimple(lr=1e-3, state_dim=state_dim, num_actions=num_actions)
   for episode in range(num_episodes):
       state = env.reset()
       done = False
       total_proc_reward = 0.0
       while not done:
           action = agent.choose_action(state, env.env)
           next_state, reward, done, _ = env.step(action)
           agent.accumulate_loss(reward)
           total_proc_reward += reward
           state = next_state
       teacher = env.env.teacher
       # For multiplication, we simply check if the produced chain exactly matches
       # the expert chain. For addition, the teacher still computes a sum.
       if env.task == "mul":
           teacher_feedback, teacher_reward = teacher.evaluate_solution(env.env.chain)
       else:
           teacher_feedback, teacher_reward = teacher.evaluate_solution(env.env.chain)
       total_final_reward = total_proc_reward + teacher_reward
       agent.finalize_episode(total_final_reward)
       if (episode + 1) % 100 == 0:
           computed = teacher.correct_answer if env.env.chain == teacher.correct_steps else "N/A"
           print(
               f"Episode {episode + 1} [{env.task}], Proc Reward: {total_proc_reward:.2f}, "
               f"Teacher Reward: {teacher_reward:.2f}, Computed: {computed}, True: {env.env.correct_answer}"
           )

   return agent
#############################################
# Simplified GUI for Multiplication Only
#############################################

class MultiplicationApp(tk.Tk):
    def __init__(self, trained_agent):
        super().__init__()
        self.title("Multiplication Solver (3-digit × 1-digit)")
        self.agent = trained_agent
        self.env = MultiplicationEnvThree()
        self.state = self.env.reset()

        self.problem_label = tk.Label(self, text=self.env.problem_text, font=("Helvetica", 16))
        self.problem_label.pack(pady=10)

        self.solution_label = tk.Label(self, text="Agent's procedure will appear here.", font=("Helvetica", 14))
        self.solution_label.pack(pady=5)

        self.result_label = tk.Label(self, text="Computed result: ", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

        self.correct_label = tk.Label(self, text="Correct answer: ", font=("Helvetica", 14))
        self.correct_label.pack(pady=5)

        self.feedback_label = tk.Label(self, text="", font=("Helvetica", 14, "bold"))
        self.feedback_label.pack(pady=5)

        self.log_text = tk.Text(self, height=12, width=80, font=("Courier", 12))
        self.log_text.pack(pady=10)

        self.solve_button = tk.Button(self, text="Solve Problem", command=self.solve_problem)
        self.solve_button.pack(pady=5)

        self.new_button = tk.Button(self, text="New Problem", command=self.new_problem)
        self.new_button.pack(pady=5)

    def new_problem(self):
        self.env.reset()
        self.state = self.env.get_state()
        self.problem_label.config(text=self.env.problem_text)
        self.correct_label.config(text=f"Correct answer: {self.env.correct_answer}")
        self.log_text.delete(1.0, tk.END)
        self.feedback_label.config(text="")
        self.solution_label.config(text="Agent's procedure will appear here.")
        self.result_label.config(text="Computed result: ")

    def solve_problem(self):
        self.log_text.delete(1.0, tk.END)
        self.env.chain = []
        self.env.chain_text = ""
        self.state = self.env.get_state()
        steps = []
        done = False
        while not done:
            action = self.agent.choose_action(self.state, self.env)
            step_text = self.env.allowed_actions[action]
            steps.append(step_text)
            self.log_text.insert(tk.END, f"Action: {step_text}\n")
            self.state, _, done, _ = self.env.step(action)
        self.solution_label.config(text=" -> ".join(steps))

        teacher = self.env.teacher
        computed_result = teacher.correct_answer if self.env.chain == teacher.correct_steps else "N/A"
        self.result_label.config(text=f"Computed result: {computed_result}")
        self.correct_label.config(text=f"Correct answer: {self.env.correct_answer}")
        feedback, _ = teacher.evaluate_solution(self.env.chain)
        self.feedback_label.config(text=feedback)
        self.problem_label.config(text=self.env.problem_text)


#############################################
# Main Execution
#############################################

if __name__ == '__main__':
    print("Training RL agent for 3-digit × 1-digit multiplication...")
    trained_agent = train_agent(num_episodes=5000)
    print("Training complete. Launching GUI.")
    app = MultiplicationApp(trained_agent)
    app.mainloop()

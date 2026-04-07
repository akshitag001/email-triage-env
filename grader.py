"""
Grader Script for Email Triage Environment
Evaluates agent performance on each task with deterministic metrics
"""

import json
import sys
from typing import Dict, List, Tuple
from env import EmailTriageEnv, Category, Priority, Action


class EmailTriageGrader:
    """Grader for evaluating email triage agents"""

    def __init__(self):
        self.results = {}

    def grade_task(self, task_name: str, agent_actions: List[Dict]) -> Dict:
        """
        Grade a single task.
        
        Args:
            task_name: 'easy', 'medium', or 'hard'
            agent_actions: List of agent actions with format:
                [
                    {"category": "support", "priority": "high", "response_required": true},
                    ...
                ]
        
        Returns:
            Dict with metrics:
            {
                "task": "easy",
                "accuracy": 0.95,
                "category_accuracy": 0.95,
                "priority_accuracy": 1.0,
                "response_accuracy": 0.9,
                "avg_score": 0.95,
                "status": "PASS" or "FAIL"
            }
        """
        env = EmailTriageEnv(task=task_name)
        env.reset()

        category_correct = 0
        priority_correct = 0
        response_correct = 0
        total_steps = 0
        total_reward = 0.0

        # Process each email in the batch
        for i, action_dict in enumerate(agent_actions):
            if i >= len(env.current_batch):
                break

            try:
                # Convert dict to Action object
                action = Action(
                    category=Category(action_dict["category"]),
                    priority=Priority(action_dict["priority"]),
                    response_required=action_dict["response_required"],
                )

                # Take step
                result = env.step(action)
                info = result.info

                # Track correctness
                if info.get("category_correct"):
                    category_correct += 1
                if info.get("priority_correct"):
                    priority_correct += 1
                if info.get("response_correct"):
                    response_correct += 1

                total_reward += result.reward
                total_steps += 1

            except Exception as e:
                print(f"Error processing action {i}: {e}")
                continue

        # Calculate metrics
        accuracy = (
            (category_correct + priority_correct + response_correct)
            / (total_steps * 3)
            if total_steps > 0
            else 0.0
        )
        category_accuracy = category_correct / total_steps if total_steps > 0 else 0.0
        priority_accuracy = priority_correct / total_steps if total_steps > 0 else 0.0
        response_accuracy = response_correct / total_steps if total_steps > 0 else 0.0
        avg_score = total_reward / total_steps if total_steps > 0 else 0.0

        # Determine pass/fail
        status = "PASS" if avg_score >= 0.6 else "FAIL"

        result = {
            "task": task_name,
            "total_steps": total_steps,
            "accuracy": round(accuracy, 3),
            "category_accuracy": round(category_accuracy, 3),
            "priority_accuracy": round(priority_accuracy, 3),
            "response_accuracy": round(response_accuracy, 3),
            "avg_score": round(avg_score, 3),
            "total_reward": round(total_reward, 2),
            "status": status,
        }

        self.results[task_name] = result
        return result

    def grade_all_tasks(
        self, agent_results: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Grade all tasks and return aggregate metrics.
        
        Args:
            agent_results: Dict mapping task names to action lists:
                {
                    "easy": [{"category": "...", "priority": "...", ...}, ...],
                    "medium": [...],
                    "hard": [...]
                }
        
        Returns:
            Dict with all results and aggregate metrics
        """
        all_results = {}
        all_scores = []

        for task in ["easy", "medium", "hard"]:
            if task in agent_results:
                result = self.grade_task(task, agent_results[task])
                all_results[task] = result
                all_scores.append(result["avg_score"])

        # Aggregate metrics
        avg_overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        all_passed = all(r["status"] == "PASS" for r in all_results.values())

        return {
            "tasks": all_results,
            "aggregate": {
                "avg_score": round(avg_overall, 3),
                "all_passed": all_passed,
                "num_tasks": len(all_results),
            },
        }

    def print_results(self):
        """Pretty print grading results"""
        print("\n" + "=" * 70)
        print("EMAIL TRIAGE GRADING RESULTS")
        print("=" * 70)

        for task, result in self.results.items():
            print(f"\nTask: {task.upper()}")
            print(f"  Status: {result['status']}")
            print(f"  Steps: {result['total_steps']}")
            print(f"  Category Accuracy: {result['category_accuracy']:.1%}")
            print(f"  Priority Accuracy: {result['priority_accuracy']:.1%}")
            print(f"  Response Accuracy: {result['response_accuracy']:.1%}")
            print(f"  Overall Accuracy: {result['accuracy']:.1%}")
            print(f"  Average Score: {result['avg_score']:.3f}/1.0")
            print(f"  Total Reward: {result['total_reward']:.2f}")

        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Example agent results (would come from inference script)
    example_results = {
        "easy": [
            {"category": "promotional", "priority": "low", "response_required": False},
            {
                "category": "support",
                "priority": "high",
                "response_required": True,
            },
            {"category": "spam", "priority": "low", "response_required": False},
        ],
        "medium": [
            {"category": "internal", "priority": "urgent", "response_required": True},
            {"category": "sales", "priority": "medium", "response_required": False},
            {
                "category": "internal",
                "priority": "medium",
                "response_required": True,
            },
            {"category": "support", "priority": "high", "response_required": True},
        ],
        "hard": [
            {"category": "internal", "priority": "urgent", "response_required": True},
            {"category": "support", "priority": "urgent", "response_required": True},
            {"category": "sales", "priority": "medium", "response_required": False},
            {
                "category": "internal",
                "priority": "medium",
                "response_required": True,
            },
            {
                "category": "internal",
                "priority": "medium",
                "response_required": True,
            },
        ],
    }

    grader = EmailTriageGrader()
    final_results = grader.grade_all_tasks(example_results)

    print(json.dumps(final_results, indent=2))
    grader.print_results()

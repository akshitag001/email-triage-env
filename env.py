"""
Email Triage & Classification Environment
Real-world task: Agents learn to categorize, prioritize, and respond to emails
"""

import json
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import random
from datetime import datetime, timedelta


class Priority(str, Enum):
    """Email priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    """Email categories"""
    SPAM = "spam"
    PROMOTIONAL = "promotional"
    SUPPORT = "support"
    SALES = "sales"
    INTERNAL = "internal"
    OTHER = "other"


class Action(BaseModel):
    """Agent action: classify and prioritize an email"""
    category: Category = Field(..., description="Email category classification")
    priority: Priority = Field(..., description="Priority level assignment")
    response_required: bool = Field(..., description="Whether response is needed")


class EmailObservation(BaseModel):
    """Email metadata for agent observation"""
    email_id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body_preview: str = Field(..., description="First 200 chars of body")
    received_time: str = Field(..., description="ISO timestamp of receipt")
    has_attachment: bool = Field(..., description="Whether email has attachments")
    current_step: int = Field(..., description="Current step in episode")
    total_emails_in_batch: int = Field(..., description="Total emails to process")


class EnvironmentReward(BaseModel):
    """Reward structure with partial credit"""
    value: float = Field(..., ge=0.0, le=1.0, description="Reward in [0, 1]")
    category_correct: bool = Field(..., description="Correct category prediction")
    priority_correct: bool = Field(..., description="Correct priority prediction")
    response_correct: bool = Field(..., description="Correct response requirement")
    efficiency_bonus: float = Field(..., ge=0.0, le=0.1, description="Bonus for quick decisions")


class StepResult(BaseModel):
    """Result of environment.step()"""
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result of environment.reset()"""
    observation: EmailObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class EmailTriageEnv:
    """Email Triage Environment for OpenEnv"""

    # Ground truth dataset
    EMAIL_DATASET = [
        {
            "id": "email_001",
            "sender": "newsletter@amazon.com",
            "subject": "Check out our new deals this week!",
            "body": "We have amazing deals on electronics, home, and more...",
            "has_attachment": False,
            "true_category": Category.PROMOTIONAL,
            "true_priority": Priority.LOW,
            "true_response": False,
        },
        {
            "id": "email_002",
            "sender": "support@stripe.com",
            "subject": "Your payment failed - action required",
            "body": "We attempted to charge your account but the payment was declined...",
            "has_attachment": True,
            "true_category": Category.SUPPORT,
            "true_priority": Priority.HIGH,
            "true_response": True,
        },
        {
            "id": "email_003",
            "sender": "boss@company.com",
            "subject": "Q4 Budget Review - Urgent",
            "body": "Please review and approve the attached Q4 budget by EOD today...",
            "has_attachment": True,
            "true_category": Category.INTERNAL,
            "true_priority": Priority.URGENT,
            "true_response": True,
        },
        {
            "id": "email_004",
            "sender": "sales@generic-pharma.com",
            "subject": "CLICK HERE for amazing opportunities!!!",
            "body": "You've been selected for an exclusive opportunity. Limited time offer...",
            "has_attachment": False,
            "true_category": Category.SPAM,
            "true_priority": Priority.LOW,
            "true_response": False,
        },
        {
            "id": "email_005",
            "sender": "team@github.com",
            "subject": "New workflow run: PR #234 failed",
            "body": "Your pull request has failed CI checks. Review: ...",
            "has_attachment": False,
            "true_category": Category.INTERNAL,
            "true_priority": Priority.MEDIUM,
            "true_response": True,
        },
        {
            "id": "email_006",
            "sender": "sales@techcorp.io",
            "subject": "Enterprise SaaS Solution - Let's Talk",
            "body": "Hi, we offer cutting-edge solutions for businesses like yours...",
            "has_attachment": True,
            "true_category": Category.SALES,
            "true_priority": Priority.MEDIUM,
            "true_response": False,
        },
        {
            "id": "email_007",
            "sender": "alerts@aws.amazon.com",
            "subject": "CRITICAL: Unusual account activity detected",
            "body": "We detected multiple failed login attempts. Review immediately...",
            "has_attachment": False,
            "true_category": Category.SUPPORT,
            "true_priority": Priority.URGENT,
            "true_response": True,
        },
        {
            "id": "email_008",
            "sender": "hr@company.com",
            "subject": "Benefits enrollment opens tomorrow",
            "body": "Annual benefits enrollment begins tomorrow. Please review your options...",
            "has_attachment": True,
            "true_category": Category.INTERNAL,
            "true_priority": Priority.MEDIUM,
            "true_response": True,
        },
        {
            "id": "email_009",
            "sender": "noreply@restaurant.local",
            "subject": "Your reservation is confirmed",
            "body": "Your reservation for 2 people on Saturday 7pm is confirmed...",
            "has_attachment": False,
            "true_category": Category.OTHER,
            "true_priority": Priority.LOW,
            "true_response": False,
        },
        {
            "id": "email_010",
            "sender": "leads@marketingagency.com",
            "subject": "Hot leads for your business",
            "body": "We have 50 qualified leads ready to close. Act now!!!",
            "has_attachment": False,
            "true_category": Category.SPAM,
            "true_priority": Priority.LOW,
            "true_response": False,
        },
    ]

    def __init__(self, task: str = "easy"):
        """
        Initialize environment.
        task: 'easy', 'medium', 'hard' - affects email mix difficulty
        """
        self.task = task
        self.current_step = 0
        self.max_steps = 10
        self.current_batch: List[Dict[str, Any]] = []
        self.batch_index = 0
        self.episode_rewards: List[float] = []
        self.step_times: List[float] = []
        self._episode_start_time = None

    def reset(self) -> ResetResult:
        """Reset environment and return initial observation"""
        self.current_step = 0
        self.batch_index = 0
        self.episode_rewards = []
        self.step_times = []
        self._episode_start_time = datetime.now()

        # Select emails based on task difficulty
        if self.task == "easy":
            # Easy: clear distinctions (spam, promo, urgent support)
            self.current_batch = [
                self.EMAIL_DATASET[1],  # Support - HIGH
                self.EMAIL_DATASET[3],  # Spam - LOW
                self.EMAIL_DATASET[0],  # Promo - LOW
            ]
        elif self.task == "medium":
            # Medium: mix of support, internal, sales
            self.current_batch = [
                self.EMAIL_DATASET[2],  # Internal - URGENT
                self.EMAIL_DATASET[5],  # Sales - MEDIUM
                self.EMAIL_DATASET[4],  # Internal - MEDIUM
                self.EMAIL_DATASET[1],  # Support - HIGH
            ]
        else:  # hard
            # Hard: ambiguous cases requiring careful analysis
            self.current_batch = [
                self.EMAIL_DATASET[2],  # Internal - URGENT
                self.EMAIL_DATASET[6],  # Support - URGENT
                self.EMAIL_DATASET[5],  # Sales - MEDIUM
                self.EMAIL_DATASET[7],  # Internal - MEDIUM
                self.EMAIL_DATASET[4],  # Internal - MEDIUM
            ]

        obs = self._get_observation()
        return ResetResult(observation=obs, info={"task": self.task})

    def step(self, action: Action) -> StepResult:
        """Process agent action and return reward"""
        if self.batch_index >= len(self.current_batch):
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"error": "Batch exhausted"},
            )

        email = self.current_batch[self.batch_index]
        self.current_step += 1
        self.batch_index += 1

        # Compute reward
        reward_obj = self._compute_reward(action, email)
        reward = reward_obj.value

        self.episode_rewards.append(reward)

        done = self.batch_index >= len(self.current_batch)

        obs = self._get_observation()

        info = {
            "email_id": email["id"],
            "category_correct": reward_obj.category_correct,
            "priority_correct": reward_obj.priority_correct,
            "response_correct": reward_obj.response_correct,
            "efficiency_bonus": reward_obj.efficiency_bonus,
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "task": self.task,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "batch_size": len(self.current_batch),
            "batch_index": self.batch_index,
            "episode_rewards": self.episode_rewards,
            "total_reward": sum(self.episode_rewards),
        }

    def _get_observation(self) -> EmailObservation:
        """Get current email observation"""
        if self.batch_index < len(self.current_batch):
            email = self.current_batch[self.batch_index]
        else:
            email = self.current_batch[-1]

        received_time = (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat()

        return EmailObservation(
            email_id=email["id"],
            sender=email["sender"],
            subject=email["subject"],
            body_preview=email["body"][:200],
            received_time=received_time,
            has_attachment=email["has_attachment"],
            current_step=self.current_step,
            total_emails_in_batch=len(self.current_batch),
        )

    def _compute_reward(self, action: Action, email: Dict[str, Any]) -> EnvironmentReward:
        """Compute detailed reward with partial credit"""
        category_correct = action.category == email["true_category"]
        priority_correct = action.priority == email["true_priority"]
        response_correct = action.response_required == email["true_response"]

        # Base reward: 0.3 per correct classification
        base_reward = 0.0
        if category_correct:
            base_reward += 0.3
        if priority_correct:
            base_reward += 0.3
        if response_correct:
            base_reward += 0.3

        # Efficiency bonus: faster processing = small bonus
        elapsed = (datetime.now() - self._episode_start_time).total_seconds()
        efficiency_bonus = min(0.1, max(0.0, 0.1 - (elapsed / 1000)))

        total_reward = min(1.0, base_reward + efficiency_bonus)

        return EnvironmentReward(
            value=total_reward,
            category_correct=category_correct,
            priority_correct=priority_correct,
            response_correct=response_correct,
            efficiency_bonus=efficiency_bonus,
        )


async def create_env(task: str = "easy") -> EmailTriageEnv:
    """Factory function for environment creation (async compatibility)"""
    return EmailTriageEnv(task=task)

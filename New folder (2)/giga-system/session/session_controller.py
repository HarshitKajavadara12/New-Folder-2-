
import sys
import time
from risk.session_guard import SessionGuard

class SessionController:
    """
    PHASE 6: The "Parent" Process
    Responsibility: Limits the runtime to a single verifiable session.
    """
    def __init__(self):
        self.guard = SessionGuard(max_loss=500.0)
        self.start_time = time.time()
        self.session_id = f"SES-{int(self.start_time)}"
        self.is_active = True

    def begin(self, equity: float):
        self.guard.initialize_account(equity)
        print(f"[SESSION] {self.session_id} STARTED. Equity: ${equity:.2f}")

    def heartbeat(self, account_equity: float):
        if not self.is_active: return False
        
        # 1. Runtime Limit (Safety)
        if time.time() - self.start_time > 36000: # 10 Hour Hard Limit
            print("[SESSION] TIME LIMIT REACHED. SHUTTING DOWN.")
            self.is_active = False
            return False
            
        # 2. Risk Guard
        if not self.guard.check_health(account_equity):
            print(f"\n{self.guard.get_status()}")
            self.stop()
            return False
            
        return True

    def stop(self):
        self.is_active = False
        print(f"[SESSION] {self.session_id} ENDED.")

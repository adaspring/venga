import hashlib
import os
from flask import session
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Simple user/password authentication manager"""
    
    def __init__(self):
        self.users = self._load_users()
        
    def _load_users(self):
        """Load users from environment variable
        Format: username1:password1,username2:password2
        """
        users = {}
        users_config = os.getenv('APP_USERS', 'admin:admin123')
        
        try:
            for user_pair in users_config.split(','):
                if ':' in user_pair:
                    username, password = user_pair.strip().split(':', 1)
                    # Store hashed password
                    users[username] = self._hash_password(password)
            
            logger.info(f"Loaded {len(users)} users")
            return users
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            # Fallback to default admin user
            return {'admin': self._hash_password('admin123')}
    
    def _hash_password(self, password):
        """Simple password hashing using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_user(self, username, password):
        """Validate username and password"""
        if username in self.users:
            return self.users[username] == self._hash_password(password)
        return False
    
    def create_guest_session(self):
        """Create a guest session without requiring credentials"""
        session["authenticated"] = True
        session["username"] = "GuestUser"
        session.permanent = True
        logger.info("Guest session created")
        return True

# Create global auth manager instance
auth_manager = AuthManager()

def get_current_user():
    """Get current logged-in username"""
    return session.get('username')
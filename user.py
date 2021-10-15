from datetime import datetime
import uuid


# User class
class User():
    def __init__(self, login, id=""):
        # Main initialiser
        self.login = login
        self.id = uuid.uuid4().hex if not id else id

    @classmethod
    def make_from_dict(cls, d):
        # Initialise User object from a dictionary
        return cls(d['login'], d['id'])

    def dict(self):
        # Return dictionary representation of the object
        return {
            "id": self.id,
            "login": self.login
        }

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return self.id


# Anonymous user class
class Anonymous():

    @property
    def is_authenticated(self):
        return False

    @property
    def is_active(self):
        return False

    @property
    def is_anonymous(self):
        return True

    def get_id(self):
        return None

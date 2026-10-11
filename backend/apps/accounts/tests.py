from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase

User = get_user_model()

STRONG_PASSWORD = "Str0ng-Pass-Phrase!"


def register_payload(**overrides):
    payload = {
        "username": "alice",
        "email": "alice@example.com",
        "password": STRONG_PASSWORD,
    }
    payload.update(overrides)
    return payload


def login(client, username="alice", password=STRONG_PASSWORD):
    return client.post(
        reverse("auth-login"), {"username": username, "password": password}, format="json"
    )


class RegisterTests(APITestCase):
    url = reverse("auth-register")

    def test_successful_registration(self):
        response = self.client.post(self.url, register_payload(), format="json")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["username"], "alice")
        self.assertEqual(response.data["email"], "alice@example.com")
        self.assertNotIn("password", response.data)
        user = User.objects.get(username="alice")
        self.assertNotEqual(user.password, STRONG_PASSWORD)  # stored hashed
        self.assertTrue(user.check_password(STRONG_PASSWORD))

    def test_invalid_registration(self):
        cases = [
            ({"username": ""}, "username"),
            ({"email": "not-an-email"}, "email"),
            ({"email": ""}, "email"),
            ({"password": ""}, "password"),
            ({"password": "short"}, "password"),
            ({"password": "password123"}, "password"),  # common password
        ]
        for overrides, field in cases:
            with self.subTest(overrides=overrides):
                response = self.client.post(self.url, register_payload(**overrides), format="json")
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.data)
        self.assertEqual(User.objects.count(), 0)

    def test_missing_fields(self):
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 400)
        for field in ("username", "email", "password"):
            self.assertIn(field, response.data)

    def test_duplicate_username(self):
        User.objects.create_user("alice", "other@example.com", STRONG_PASSWORD)
        response = self.client.post(self.url, register_payload(), format="json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("username", response.data)

    def test_duplicate_email_case_insensitive(self):
        User.objects.create_user("bob", "alice@example.com", STRONG_PASSWORD)
        response = self.client.post(
            self.url, register_payload(email="ALICE@example.com"), format="json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("email", response.data)


class LoginTests(APITestCase):
    def setUp(self):
        User.objects.create_user("alice", "alice@example.com", STRONG_PASSWORD)

    def test_successful_login(self):
        response = login(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)
        self.assertIn("refresh", response.data)

    def test_invalid_login(self):
        for username, password in (("alice", "wrong-password"), ("nobody", STRONG_PASSWORD)):
            with self.subTest(username=username):
                response = login(self.client, username, password)
                self.assertEqual(response.status_code, 401)
                self.assertNotIn("access", response.data)


class MeTests(APITestCase):
    url = reverse("auth-me")

    def setUp(self):
        self.user = User.objects.create_user("alice", "alice@example.com", STRONG_PASSWORD)

    def test_authenticated_me(self):
        access = login(self.client).data["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data,
            {"id": self.user.id, "username": "alice", "email": "alice@example.com"},
        )

    def test_unauthenticated_me(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 401)

    def test_invalid_token_me(self):
        self.client.credentials(HTTP_AUTHORIZATION="Bearer not-a-real-token")
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 401)


class TokenRefreshTests(APITestCase):
    url = reverse("auth-token-refresh")

    def setUp(self):
        User.objects.create_user("alice", "alice@example.com", STRONG_PASSWORD)
        self.tokens = login(self.client).data

    def test_refresh_returns_new_access_token(self):
        response = self.client.post(self.url, {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)

    def test_refresh_with_invalid_token(self):
        response = self.client.post(self.url, {"refresh": "garbage"}, format="json")
        self.assertEqual(response.status_code, 401)

    def test_access_token_cannot_be_used_to_refresh(self):
        response = self.client.post(self.url, {"refresh": self.tokens["access"]}, format="json")
        self.assertEqual(response.status_code, 401)

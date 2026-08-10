import uuid


def _signup_payload(**overrides):
    payload = {
        "user_id": f"testuser_{uuid.uuid4().hex[:10]}",
        "nickname": "테스트유저",
        "password": "password123",
        "gender": "man",
        "birth_date": "1998-01-01",
        "height": 175,
        "weight": 68,
    }
    payload.update(overrides)
    return payload


def _signup_and_login(client):
    payload = _signup_payload()
    signup_res = client.post("/user/signup", json=payload)
    assert signup_res.status_code == 200

    login_res = client.post(
        "/user/login", json={"user_id": payload["user_id"], "password": payload["password"]}
    )
    assert login_res.status_code == 200
    token = login_res.json()["data"]["access_token"]
    return payload, token


def test_signup_success(client):
    payload = _signup_payload()
    res = client.post("/user/signup", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["user_id"] == payload["user_id"]
    assert body["data"]["nickname"] == payload["nickname"]
    assert body["data"]["gender"] == "man"
    assert body["data"]["is_calibrated"] is False
    # 비밀번호 해시가 응답에 노출되면 안 된다.
    assert "password" not in body["data"]
    assert "password_hash" not in body["data"]


def test_signup_duplicate_user_id_fails(client):
    payload = _signup_payload()
    first = client.post("/user/signup", json=payload)
    assert first.status_code == 200

    second = client.post("/user/signup", json=payload)
    assert second.status_code == 400
    assert second.json()["status"] == "error"


def test_login_success(client):
    payload = _signup_payload()
    client.post("/user/signup", json=payload)

    res = client.post(
        "/user/login", json={"user_id": payload["user_id"], "password": payload["password"]}
    )

    assert res.status_code == 200
    body = res.json()
    assert body["data"]["token_type"] == "bearer"
    assert body["data"]["access_token"]
    assert body["data"]["user"]["user_id"] == payload["user_id"]


def test_login_wrong_password_fails(client):
    payload = _signup_payload()
    client.post("/user/signup", json=payload)

    res = client.post("/user/login", json={"user_id": payload["user_id"], "password": "wrongpassword"})

    assert res.status_code == 401
    assert res.json()["status"] == "error"


def test_login_nonexistent_user_fails(client):
    res = client.post("/user/login", json={"user_id": "no_such_user_xyz", "password": "whatever123"})

    assert res.status_code == 401


def test_get_me_requires_token(client):
    res = client.get("/user/me")

    assert res.status_code == 401


def test_get_me_with_valid_token(client):
    payload, token = _signup_and_login(client)

    res = client.get("/user/me", headers={"Authorization": f"Bearer {token}"})

    assert res.status_code == 200
    body = res.json()["data"]
    assert body["user_id"] == payload["user_id"]
    assert body["height"] == payload["height"]
    assert body["weight"] == payload["weight"]


def test_update_me_partial_update(client):
    _, token = _signup_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    res = client.patch("/user/me", json={"height": 180, "weight": 72}, headers=headers)

    assert res.status_code == 200
    body = res.json()["data"]
    assert body["height"] == 180
    assert body["weight"] == 72

    # 안 보낸 필드(gender)는 원래 값 그대로 유지돼야 한다.
    me_res = client.get("/user/me", headers=headers)
    assert me_res.json()["data"]["gender"] == "man"


def test_update_me_requires_token(client):
    res = client.patch("/user/me", json={"height": 180})

    assert res.status_code == 401


def test_delete_me_then_login_fails(client):
    payload, token = _signup_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}

    delete_res = client.delete("/user/me", headers=headers)
    assert delete_res.status_code == 200
    assert delete_res.json()["data"]["message"]

    # 탈퇴 후 같은 계정으로 로그인은 실패해야 한다.
    login_res = client.post(
        "/user/login", json={"user_id": payload["user_id"], "password": payload["password"]}
    )
    assert login_res.status_code == 401


def test_delete_me_requires_token(client):
    res = client.delete("/user/me")

    assert res.status_code == 401

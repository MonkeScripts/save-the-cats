# `main.cpp` — Long-Run Stall Fixes

ESP32 zenoh-pico TLS publisher was stopping after long uptime. Likely causes: stale TLS link after WiFi blip, BLOCK congestion + ESP32 TLS BIO send bug, slow heap leak from `z_publisher_put` failure path, NTP drift breaking TLS cert validity. Below are the seven changes applied to `main.cpp`.

---

## 1. New globals (state for recovery + watchdog)

Added near other static state:

```cpp
// Stall-recovery state
static int consec_fail = 0;
static uint32_t last_ntp_ms = 0;
static uint32_t last_heap_ms = 0;
static const uint32_t NTP_RESYNC_INTERVAL_MS = 3600000UL; // 1h
static const uint32_t HEAP_LOG_INTERVAL_MS  = 10000UL;    // 10s
static const int CONSEC_FAIL_THRESHOLD = 20;

// Forward decls
bool bringUpZenoh();
void teardownZenoh();
```

---

## 2. JSON doc consistency in `data_handler`

**Before:**
```cpp
JsonDocument doc;
```
**After:**
```cpp
StaticJsonDocument<256> doc;
```
Removes dynamic alloc inside subscriber callback (fragmentation source).

---

## 3. New `teardownZenoh()` + `bringUpZenoh()` helpers

Refactored zenoh setup so it can be re-run after WiFi reconnect or watchdog.

```cpp
void teardownZenoh() {
    z_undeclare_subscriber(z_subscriber_move(&sub));
    z_undeclare_publisher(z_publisher_move(&pub));
    z_session_drop(z_session_move(&s));
}

bool bringUpZenoh() {
    z_owned_config_t config;
    z_config_default(&config);
    zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_MODE_KEY, MODE);
    if (strcmp(LOCATOR, "") != 0) {
        if (strcmp(MODE, "client") == 0) {
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_CONNECT_KEY, LOCATOR);
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_ROOT_CA_CERTIFICATE_BASE64_KEY, my_root_ca);
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_ENABLE_MTLS_KEY, "false");
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_VERIFY_NAME_ON_CONNECT_KEY, "false");
        } else {
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_LISTEN_KEY, LOCATOR);
        }
    }

    if (z_open(&s, z_config_move(&config), NULL) < 0) return false;

    if (zp_start_read_task(z_session_loan_mut(&s), NULL) < 0 ||
        zp_start_lease_task(z_session_loan_mut(&s), NULL) < 0) {
        z_session_drop(z_session_move(&s));
        return false;
    }

    z_view_keyexpr_t ke;
    z_view_keyexpr_from_str_unchecked(&ke, KEYEXPRPUB);

    // *** Key fix: DROP congestion control prevents BLOCK starvation
    //     if ESP32 TLS TX mutex stalls. ***
    z_publisher_options_t pub_opts;
    z_publisher_options_default(&pub_opts);
    pub_opts.congestion_control = Z_CONGESTION_CONTROL_DROP;
    if (z_declare_publisher(z_session_loan(&s), &pub, z_view_keyexpr_loan(&ke), &pub_opts) < 0) return false;

    z_owned_closure_sample_t callback;
    z_closure_sample(&callback, data_handler, NULL, NULL);
    z_view_keyexpr_from_str_unchecked(&ke, KEYEXPRSUB);
    if (z_declare_subscriber(z_session_loan(&s), &sub, z_view_keyexpr_loan(&ke),
                             z_closure_sample_move(&callback), NULL) < 0) return false;
    return true;
}
```

`setup()` now collapses the old ~100-line zenoh init block into:
```cpp
if (!bringUpZenoh()) {
    Serial.println("Initial Zenoh bringup failed; restarting");
    delay(2000);
    ESP.restart();
}
last_ntp_ms = millis();
```

---

## 4. Rebuild Zenoh on WiFi reconnect (`maintainWiFi`)

The original code only marked `wifi_was_connected = true` after reconnect — the underlying TLS link was already dead, so subsequent `z_publisher_put` calls silently no-op'd.

**Before:**
```cpp
if (status == WL_CONNECTED) {
    if (!wifi_was_connected) {
        Serial.print("WiFi reconnected. IP: ");
        Serial.println(WiFi.localIP());
        oledStatus("IMU WORKOUT SYS", "WiFi reconnected");
        wifi_was_connected = true;
    }
    return;
}
```

**After:**
```cpp
if (status == WL_CONNECTED) {
    if (!wifi_was_connected) {
        Serial.print("WiFi reconnected. IP: ");
        Serial.println(WiFi.localIP());
        oledStatus("IMU WORKOUT SYS", "Rebuilding Zenoh");

        // Underlying TLS link is dead after WiFi drop; rebuild full session.
        teardownZenoh();
        delay(500);
        syncTime();
        if (!bringUpZenoh()) {
            Serial.println("Zenoh rebuild failed; restarting");
            delay(1000);
            ESP.restart();
        }
        consec_fail = 0;
        last_ntp_ms = millis();
        wifi_was_connected = true;
    }
    return;
}
```

This is the single biggest fix for the stall.

---

## 5. Periodic NTP resync (top of `loop()`)

ESP32 RTC drifts a few seconds/hour. After long uptime the local clock can fall outside the TLS cert validity window, breaking any reconnect handshake.

```cpp
uint32_t now_tick = millis();
if (now_tick - last_ntp_ms > NTP_RESYNC_INTERVAL_MS) {
    last_ntp_ms = now_tick;
    configTime(0, 0, "pool.ntp.org", "time.google.com", "time.windows.com");
}
```

---

## 6. Heap log + put-failure watchdog

Added in `loop()`:

```cpp
// Heap visibility for leak detection
if (now_tick - last_heap_ms > HEAP_LOG_INTERVAL_MS) {
    last_heap_ms = now_tick;
    Serial.printf("[health] heap=%u consec_fail=%d\n",
                  (unsigned)ESP.getFreeHeap(), consec_fail);
}

// Watchdog: rebuild session if puts keep failing
if (consec_fail > CONSEC_FAIL_THRESHOLD) {
    Serial.println("[watchdog] consec_fail exceeded; rebuilding Zenoh");
    teardownZenoh();
    delay(500);
    if (!bringUpZenoh()) {
        Serial.println("[watchdog] rebuild failed; restarting");
        delay(1000);
        ESP.restart();
    }
    consec_fail = 0;
    return;
}
```

---

## 7. Drop owned objects on `z_publisher_put` failure

`z_publisher_put` only consumes `payload`/`encoding` on success. Each failure leaked them — over hours, heap fragmented and the session eventually died silently.

**Before:**
```cpp
if (z_publisher_put(z_publisher_loan(&pub), z_bytes_move(&payload), &options) < 0) {
    Serial.println("Error while publishing sample data");
}
```

**After:**
```cpp
if (z_publisher_put(z_publisher_loan(&pub), z_bytes_move(&payload), &options) < 0) {
    // Drop owned objects on failure to prevent slow heap leak
    z_bytes_drop(z_bytes_move(&payload));
    z_encoding_drop(z_encoding_move(&options.encoding));
    consec_fail++;
    Serial.printf("Error while publishing sample data (consec_fail=%d)\n", consec_fail);
} else {
    consec_fail = 0;
}
```

---

## Summary of Effects

| # | Fix | Addresses |
|---|-----|-----------|
| 1 | New globals | infra for 4–7 |
| 2 | StaticJsonDocument in `data_handler` | callback heap fragmentation |
| 3 | `bringUpZenoh` / `teardownZenoh` | reuse setup path |
| 3 | `Z_CONGESTION_CONTROL_DROP` on publisher | BLOCK starvation under TLS BIO bug |
| 4 | Rebuild Zenoh on WiFi reconnect | **stale TLS link silent stall (primary cause)** |
| 5 | Periodic NTP resync | TLS cert validity drift |
| 6 | Heap log + watchdog | self-recovery + observability |
| 7 | Drop payload/encoding on put failure | slow heap leak |

## Verification

1. PlatformIO build + flash ESP32.
2. Start router + computer subscriber on `esp/**`. Confirm `esp/chest` JSON @10Hz.
3. Soak 12–24h: monitor `[health] heap=... consec_fail=...` log line — heap should stay flat.
4. Fault inject:
   - Toggle WiFi AP off ~30s, on. Expect: stall during outage, `Rebuilding Zenoh` message, stream resumes.
   - Briefly block router TCP. Expect: `consec_fail` climbs, watchdog rebuilds session, recovery.

//
// Copyright (c) 2022 ZettaScale Technology
//
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License 2.0 which is available at
// http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
// which is available at https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
//
// Contributors:
//   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
//

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <time.h>
#include <math.h>

#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_SSD1306.h>
#include <zenoh-pico.h>

#include "secrets.h"

// =====================================================
// OLED display parameters
// =====================================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1

// =====================================================
// IMU / sampling parameters
// =====================================================
static const uint32_t SAMPLE_DELAY_MS = 110;   // 10 Hz sampling
static const int FS_HZ = 50;
static const uint32_t OLED_PERIOD_MS = 500;    // 2 Hz OLED refresh
static uint32_t last_oled_ms = 0;
static bool oled_ok = false;

// WiFi Reconnection
static uint32_t last_wifi_retry_ms = 0;
static const uint32_t WIFI_RETRY_INTERVAL_MS = 3000;  // retry every 3s
static bool wifi_was_connected = false;

// =====================================================
// Zenoh configuration
// =====================================================
#if Z_FEATURE_PUBLICATION == 1 && Z_FEATURE_LINK_TLS == 1

// Client mode values
#define MODE "client"
#define LOCATOR "tls/172.20.10.3:7447" //shaoliang
// #define LOCATOR "tls/172.20.10.9:7447" //TimL
// #define LOCATOR "tls/192.168.0.93:7447"   // alternative locator
// Peer mode values
// #define MODE "peer"
// #define LOCATOR "udp/224.0.0.225:7447#iface=en0"

#define KEYEXPRPUB "esp/chest"
#define KEYEXPRSUB "computer/**"
#define VALUE "[ARDUINO]{ESP32} Publication from Zenoh-Pico!"

// =====================================================
// Global objects
// =====================================================
z_owned_session_t s;
z_owned_publisher_t pub;
z_owned_subscriber_t sub;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Adafruit_MPU6050 mpu;

static int idx = 0;
static int action = 0;

void syncTime() {
    // 1. Try multiple NTP servers
    // pool.ntp.org is standard, but some networks prefer local ones
    configTime(0, 0, "pool.ntp.org", "time.google.com", "time.windows.com");

    Serial.println("Waiting for NTP time sync...");
    time_t now = time(nullptr);
    int retry = 0;

    // 1704067200 is Jan 1, 2024. We wait until we pass this.
    while (now < 1704067200 && retry < 20) {
        delay(1000);
        now = time(nullptr);
        Serial.printf("Current Epoch: %ld (Waiting for 2024+)\n", (long)now);
        retry++;
    }

    if (now < 1704067200) {
        Serial.println("NTP Failed! TLS will likely fail. Check if port 123 is blocked.");
    } else {
        struct tm timeinfo;
        gmtime_r(&now, &timeinfo);
        Serial.print("Time synchronized: ");
        Serial.println(asctime(&timeinfo));
    }
}
void data_handler(z_loaned_sample_t* sample, void* arg) {
    z_view_string_t keystr;
    z_keyexpr_as_view_string(z_sample_keyexpr(sample), &keystr);
    z_owned_string_t value;
    z_bytes_to_string(z_sample_payload(sample), &value);
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, z_string_data(z_string_loan(&value)));
    if (error) {
        Serial.print(F("deserializeJson() failed: "));
        Serial.println(error.c_str());
        z_string_drop(z_string_move(&value));
        return;
    }
    action = doc["action"];

    Serial.print(" >> [Subscription listener] Received (");
    Serial.write(z_string_data(z_view_string_loan(&keystr)), z_string_len(z_view_string_loan(&keystr)));
    Serial.print(", ");
    Serial.write(z_string_data(z_string_loan(&value)), z_string_len(z_string_loan(&value)));
    Serial.println(")");
    Serial.println("    Parsed action: " + String(action));

    z_string_drop(z_string_move(&value));
}

void oledStatus(const char* line1, const char* line2) {
    if (!oled_ok) return;  // <-- prevents crash if OLED not ready
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.println(line1);
    display.println(line2);
    display.display();
}

void connectWiFiBlocking() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.println("SSID: " + String(WIFI_SSID));
    Serial.print("Connecting to WiFi");

    oledStatus("IMU WORKOUT SYS", "Waiting for WiFi");

    uint32_t last_anim = 0;
    int dots = 0;

    while (WiFi.status() != WL_CONNECTED) {
        uint32_t now = millis();

        if (now - last_anim > 500) {
            last_anim = now;
            dots = (dots + 1) % 4;

            char msg[22];
            snprintf(msg, sizeof(msg), "Waiting WiFi%s",
                     dots == 0 ? "" :
                     dots == 1 ? "." :
                     dots == 2 ? ".." : "...");

            oledStatus("IMU WORKOUT SYS", msg);
            Serial.print(".");
        }

        delay(200);
    }

    Serial.println();
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());

    oledStatus("IMU WORKOUT SYS", "WiFi connected");
    delay(1000);

    wifi_was_connected = true;
}

void maintainWiFi() {
    wl_status_t status = WiFi.status();

    if (status == WL_CONNECTED) {
        if (!wifi_was_connected) {
            Serial.print("WiFi reconnected. IP: ");
            Serial.println(WiFi.localIP());
            oledStatus("IMU WORKOUT SYS", "WiFi reconnected");
            wifi_was_connected = true;
        }
        return;
    }

    if (wifi_was_connected) {
        Serial.println("WiFi lost. Attempting reconnect...");
        oledStatus("IMU WORKOUT SYS", "WiFi lost");
        wifi_was_connected = false;
    }

    uint32_t now = millis();
    if (now - last_wifi_retry_ms >= WIFI_RETRY_INTERVAL_MS) {
        last_wifi_retry_ms = now;

        Serial.println("Retrying WiFi...");
        oledStatus("IMU WORKOUT SYS", "Reconnecting WiFi");

        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
}

void setup() {
    // Initialize Serial for debug
    Serial.begin(115200);
    while (!Serial) {
        delay(1000);
    }
    Serial.println("Starting Zenoh-Pico Arduino ESP32 example...");
    // ---- I2C + OLED init MUST happen before any oledStatus() calls ----
    Wire.begin();  // or Wire.begin(21, 22);

    oled_ok = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    if (!oled_ok) {
        Serial.println("OLED init failed (0x3C).");
    } else {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("IMU WORKOUT SYS");
        display.println("Booting...");
        display.display();
        delay(500);
    }

    // ==== WiFi Connection ===
    connectWiFiBlocking();

    syncTime();

    // Initialize Zenoh Session and other parameters
    z_owned_config_t config;
    // Creates a default config
    z_config_default(&config);
    // Insert mode changes based on config key
    zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_MODE_KEY, MODE);
    if (strcmp(LOCATOR, "") != 0) {
        // If mode == client
        if (strcmp(MODE, "client") == 0) {
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_CONNECT_KEY, LOCATOR);
            if (zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_ROOT_CA_CERTIFICATE_BASE64_KEY, my_root_ca) != Z_OK) {
                Serial.println("Failed to set inline CA certificate");
            } else {
                Serial.println("Set inline CA certificate");
            }
            if (zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_ENABLE_MTLS_KEY, "false") != Z_OK) {
                fprintf(stderr, "Failed to Disable mTLS\n");
            } else {
                Serial.println("Disabled mTLS");
            }
            if (zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_TLS_VERIFY_NAME_ON_CONNECT_KEY, "false") != Z_OK) {
                fprintf(stderr, "Failed to Disable name verification\n");
            } else {
                Serial.println("Disabled name verification");
            }
        } else {
            zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_LISTEN_KEY, LOCATOR);
        }
    }

    // Open Zenoh session
    Serial.print("Opening Zenoh Session...");
    if (z_open(&s, z_config_move(&config), NULL) < 0) {
        Serial.println("Unable to open session!");
        while (1) {
            // Serial.println("Stuck and unable to open Zenoh Session!");
        }
    }
    Serial.println("OK");
    /*
    Read and Lease Tasks Explanation
    These are background tasks that keep your Zenoh communication session alive and responsive:

    Read Task
    Continuously monitors incoming messages from the Zenoh network and processes them. Think of it as a "listener" that:

    Checks for new data/commands arriving from other publishers or subscribers
    Processes those messages according to your subscriptions
    Prevents your device from missing network events
    Lease Task
    Maintains your device's "heartbeat" on the network. It:

    Periodically sends keepalive signals to the Zenoh router/peers
    Tells the network "I'm still here and active"
    Prevents the router from timing out your connection and dropping you
    Why Both Matter
    In the context of embedded systems (like Arduino), these tasks run asynchronously:

    Without read task: You'd never receive incoming messages
    Without lease task: The network would think you're offline and disconnect you
    */

    // Start read and lease tasks for zenoh-pico
    if (zp_start_read_task(z_session_loan_mut(&s), NULL) < 0 || zp_start_lease_task(z_session_loan_mut(&s), NULL) < 0) {
        Serial.println("Unable to start read and lease tasks\n");
        z_session_drop(z_session_move(&s));
        while (1) {
            // Serial.println("Stuck and unable to start read and lease tasks\n");
        }
    }

    // Declare Zenoh publisher
    Serial.print("Declaring publisher for ");
    Serial.print(KEYEXPRPUB);
    Serial.println("...");
    z_view_keyexpr_t ke;
    z_view_keyexpr_from_str_unchecked(&ke, KEYEXPRPUB);
    if (z_declare_publisher(z_session_loan(&s), &pub, z_view_keyexpr_loan(&ke), NULL) < 0) {
        Serial.println("Unable to declare publisher for key expression!");
        while (1) {
            ;
        }
    }

    // Declare Zenoh subscriber
    Serial.print("Declaring Subscriber on ");
    Serial.print(KEYEXPRSUB);
    Serial.println(" ...");
    z_owned_closure_sample_t callback;
    z_closure_sample(&callback, data_handler, NULL, NULL);
    z_view_keyexpr_from_str_unchecked(&ke, KEYEXPRSUB);
    if (z_declare_subscriber(z_session_loan(&s), &sub, z_view_keyexpr_loan(&ke), z_closure_sample_move(&callback),
                             NULL) < 0) {
        Serial.println("Unable to declare subscriber.");
        while (1) {
            ;
        }
    }
    Serial.println("OK");
    Serial.println("Zenoh setup finished!");

    // MPU6050 init
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip. Check wiring!");
        while (1) delay(1000);
    }
    Serial.println("MPU6050 Found!");

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    delay(300);
}

void loop() {
    delay(SAMPLE_DELAY_MS);  // adjusting frequency sampling
    maintainWiFi();

    if (WiFi.status() != WL_CONNECTED) {
        delay(50);   // small pause while reconnecting
        return;      // do not sample/publish until WiFi is back
    }
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    float ax = a.acceleration.x;
    float ay = a.acceleration.y;
    float az = a.acceleration.z;
    float avm = sqrtf(ax * ax + ay * ay + az * az);
    uint32_t now_ms = millis();

    // Gyroscope data
    float gx = g.gyro.x;
    float gy = g.gyro.y;
    float gz = g.gyro.z;

    //Create JSON payload
    StaticJsonDocument<128> doc;
    //doc["fs_hz"] = FS_HZ;
    doc["t_ms"] = now_ms;
    doc["ax"] = ax;
    doc["ay"] = ay;
    doc["az"] = az;
    doc["avm"] = avm;
    doc["gx"] = gx;
    doc["gy"] = gy;
    doc["gz"] = gz;

    char json_buf[256];
    size_t n = serializeJson(doc, json_buf, sizeof(json_buf));
    if (n == 0) {
        Serial.println("Error: json_buf too small");
        return;
    }

    //Serial output
    // Serial.printf(
    //     "t=%lu ax=%.2f ay=%.2f az=%.2f avm=%.2f gx=%.2f gy=%.2f gz=%.2f\n",
    //     (unsigned long)now_ms, ax, ay, az, avm, gx, gy, gz
    // );

    // ===== Zenoh publish (same as before) =====
    z_owned_bytes_t payload;
    z_bytes_copy_from_str(&payload, json_buf);

    z_owned_encoding_t encoding;
    z_encoding_from_str(&encoding, "text/json");

    z_publisher_put_options_t options;
    z_publisher_put_options_default(&options);
    options.encoding = z_encoding_move(&encoding);

    if (z_publisher_put(z_publisher_loan(&pub), z_bytes_move(&payload), &options) < 0) {
        Serial.println("Error while publishing sample data");
    }
    delay(1);
    if (millis() - last_oled_ms > OLED_PERIOD_MS) {
        last_oled_ms = millis();

        oledStatus("IMU WORKOUT SYS", "Streaming Data");
    }
}
#else
void setup() {
    Serial.println("ERROR: Zenoh pico was compiled without Z_FEATURE_PUBLICATION but this example requires it.");
    return;
}
void loop() {}
#endif

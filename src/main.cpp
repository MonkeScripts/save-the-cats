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
#include <ArduinoJson.h>
#include <WiFi.h>
#include <time.h>
#include <zenoh-pico.h>

#include "secrets.h"

#if Z_FEATURE_PUBLICATION == 1 and Z_FEATURE_LINK_TLS == 1

// Client mode values (comment/uncomment as needed)
#define MODE "client"
#define LOCATOR "tls/192.168.0.93:7447"  // If empty, it will scout
// Peer mode values (comment/uncomment as needed)
// #define MODE "peer"
// #define LOCATOR "udp/224.0.0.225:7447#iface=en0"

#define KEYEXPRPUB "esp/imu1"
#define KEYEXPRSUB "computer/**"
#define VALUE "[ARDUINO]{ESP32} Publication from Zenoh-Pico!"

z_owned_session_t s;
z_owned_publisher_t pub;
z_owned_subscriber_t sub;
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

void setup() {
    // Initialize Serial for debug
    Serial.begin(115200);
    while (!Serial) {
        delay(1000);
    }
    Serial.println("Starting Zenoh-Pico Arduino ESP32 example...");

    // Set WiFi in STA mode and trigger attachment
    Serial.print("Connecting to WiFi: ...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.println("SSID: " + String(WIFI_SSID));
    Serial.printf("Password: %s\n", WIFI_PASS);
    Serial.printf("ROOT CA %s\n", my_root_ca);
    while (WiFi.status() != WL_CONNECTED) {
        Serial.println("Attempting to connect to WiFi...");
        delay(1000);
    }
    Serial.println(WiFi.localIP());
    Serial.println("OK");
    // syncTime();

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

    delay(300);
}

void loop() {
    delay(500);
    // adapted from https://registry.platformio.org/libraries/bblanchon/ArduinoJson
    JsonDocument doc;
    doc["ax"] = 0.1f;
    doc["ay"] = 0.2f;
    doc["az"] = 0.3f;
    doc["gx"] = 0.4f;
    doc["gy"] = 0.5f;
    doc["gz"] = 0.6f;
    doc["action"] = action;
    char json_buf[4096];
    serializeJson(doc, json_buf);

    if (strlen(json_buf) >= sizeof(json_buf)) {
        Serial.println("Error: length of payload exceeds allocated json_buf");
        return;
    }

    Serial.print("Writing Data ('");
    Serial.print(KEYEXPRPUB);
    Serial.print("': '");
    Serial.print(json_buf);
    Serial.println("')");

    z_owned_bytes_t payload;
    z_bytes_copy_from_str(&payload, json_buf);

    z_owned_encoding_t encoding;
    z_encoding_from_str(&encoding, "text/json");

    z_publisher_put_options_t options;
    z_publisher_put_options_default(&options);
    options.encoding = z_encoding_move(&encoding);

    if (z_publisher_put(z_publisher_loan(&pub), z_bytes_move(&payload), &options) < 0) {
        Serial.println("Error while publishing data");
    }
}
#else
void setup() {
    Serial.println("ERROR: Zenoh pico was compiled without Z_FEATURE_PUBLICATION but this example requires it.");
    return;
}
void loop() {}
#endif


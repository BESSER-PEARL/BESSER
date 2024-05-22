/**
* Copyright 2024 Google LLC
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

# [START gke_quickstart_autopilot_app]

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.default.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.default.master_auth[0].cluster_ca_certificate)

  ignore_annotations = [
    "^autopilot\\.gke\\.io\\/.*",
    "^cloud\\.google\\.com\\/.*"
  ]
}


resource "kubernetes_deployment_v1" "default-" {
  metadata {
    name = "deployment1" 
    labels = {
      app = "app1"       }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "app1"         }
    }

    template {
      metadata {
        labels = {
          app = "app1"           }
      }

      spec {
        container {
          image = "image/latest"
          name  = "container1"

          resources {
            limits = {
              cpu    = "500"
              memory = "512Mi"
            }
            requests = {
              cpu    = "10m"
              memory = "100Mi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "default-1" {
  metadata {
    name = "service1"
  }

  spec {
    selector = {
      app = "app1"
    }
    port {
      port        = 80
      target_port = 8000
    }

    type = "LoadBalancer"
  }

  depends_on = [time_sleep.wait_service_cleanup]
}


# Provide time for Service cleanup
resource "time_sleep" "wait_service_cleanup" {
  depends_on = [google_container_cluster.default]

  destroy_duration = "180s"
}
# [END gke_quickstart_autopilot_app]
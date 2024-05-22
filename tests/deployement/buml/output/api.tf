resource "google_project_service" "compute" {
   project="neon-nexus-422908-u9"
    service = "compute.googleapis.com"
  }
  
  resource "google_project_service" "kubernetes" {
    project="neon-nexus-422908-u9"
    service = "container.googleapis.com"
  }
  resource "google_project_service" "anthos" {
    project="neon-nexus-422908-u9"
    service = "anthos.googleapis.com"
  }
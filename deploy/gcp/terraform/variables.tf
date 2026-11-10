variable "project_id" {
  type        = string
  description = "GCP project ID that will own the VM and firewall rules."
}

variable "region" {
  type        = string
  default     = "europe-west1"
  description = "GCP region. europe-west1 (Belgium) gives low RTT to Greece; switch if your pilots are elsewhere."
}

variable "zone" {
  type        = string
  default     = "europe-west1-b"
  description = "Zone inside the region for the VM."
}

variable "letsencrypt_email" {
  type        = string
  description = "Email Caddy registers with Let's Encrypt for expiry notifications. Any email you control; doesn't need to be Gmail."
}

variable "machine_type" {
  type        = string
  default     = "e2-small"
  description = "VM size. e2-micro (~$7/mo) is enough for tunnel + control; e2-small (~$14/mo) gives more headroom for multiple concurrent MJPEG viewers."
}

variable "basic_auth_user" {
  type        = string
  default     = "pilot"
  description = "Username for the dashboard/drone-UI login prompt."
}

variable "basic_auth_bcrypt" {
  type        = string
  description = "Bcrypt hash of the dashboard password. Generate with: docker run --rm caddy:2 caddy hash-password -p 'your-password'"
  sensitive   = true
}

variable "frp_token" {
  type        = string
  description = "Shared secret Jetsons present to the frp server. Generate with: openssl rand -hex 32"
  sensitive   = true
}

variable "frp_admin_user" {
  type        = string
  default     = "admin"
  description = "Username for the frp server admin API (used by the dashboard to list online drones)."
}

variable "frp_admin_password" {
  type        = string
  description = "Password for the frp admin API. Generate with: openssl rand -hex 16"
  sensitive   = true
}

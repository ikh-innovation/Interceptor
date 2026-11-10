output "vm_public_ip" {
  value       = google_compute_address.drones.address
  description = "Public IP of the relay VM."
}

output "domain" {
  value       = local.domain
  description = "Computed sslip.io hostname for the relay. Resolves to vm_public_ip with no DNS setup."
}

output "dashboard_url" {
  value       = "https://${local.domain}/"
  description = "Open this on your phone. First visit takes ~15s while Caddy fetches a Let's Encrypt cert."
}

output "jetson_install_hint" {
  value = "sudo deploy/gcp/jetson/install-frpc.sh --name <jetson-name> --server ${local.domain} --token <frp_token>"
  description = "Run this on each Jetson after `git pull`. Token is var.frp_token from your terraform.tfvars."
}

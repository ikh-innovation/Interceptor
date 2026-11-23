# Ground Planner: TEST και ΕΛΚΑΚ

```
LAPTOP ──RFD──► DRONE (Jetson)      στόχοι + εντολές
LAPTOP ◄─RFD─── DRONE               θέση + κατάσταση του drone
```

- **Laptop ↔ drone: μόνο RFD.** Δεν χρειάζεται Wi-Fi.
- **Internet μόνο από πριν**, για να αποθηκευτεί ο χάρτης.
- **Το drone ακολουθεί τον στόχο συνέχεια, 10 m από κάτω του.** Δεν κάνει ENGAGE, εκτός αν τσεκάρεις «AUTO ENGAGE».
- **Ο πιλότος με τον πομπό μπορεί πάντα να πάρει τον έλεγχο.**

---

# 1. TEST (ψεύτικοι στόχοι που σχεδιάζεις εσύ)

Χρειάζεσαι: laptop, RFD στο laptop, drone με RFD στο Jetson.

### Βήμα 1: Μία φορά, με internet

```bash
cd ~/Desktop/dev/Interceptor
git pull
sudo usermod -aG dialout $USER     # μετά κάνε log out και log in
```

### Βήμα 2: Ο χάρτης (μία φορά ανά περιοχή, με internet)

```bash
python3 src/ground_planner.py --port 9002
```

1. Άνοιξε `http://localhost:9002`.
2. Πήγαινε τον χάρτη εκεί που θα πετάξεις.
3. Πάτα **«Cache map here»** και περίμενε να γεμίσει (π.χ. `900/900`).
4. Κλείσε με `Ctrl+C`.

### Βήμα 3: Το drone

1. Το RFD του drone είναι στο **USB του Jetson**.
2. Ανάβεις το drone. Η εφαρμογή ξεκινά μόνη της.
3. Μόνο αν άλλαξε κώδικας, στο Jetson:
   ```bash
   cd ~/Documents/Interceptor && git pull && sudo systemctl restart wasp-tracker
   ```

### Βήμα 4: Το RFD του laptop

Βάζεις το RFD στο USB του laptop και περιμένεις το **πράσινο φως σταθερό**.

### Βήμα 5: Ξεκινάς

```bash
cd ~/Desktop/dev/Interceptor
python3 src/ground_planner.py --port 9002 \
    --rfd-dev /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG013VP7-if00-port0
```

1. Άνοιξε `http://localhost:9002`.
2. Πάνω δεξιά πρέπει να δεις **Drone: linked (RFD 0.5s)**.

### Βήμα 6: Πετάς

1. Καρτέλα **TEST**: σχεδίασε στόχο (Circle, Line ή Draw).
   - Το **Altitude** είναι **ύψος πάνω από το σημείο απογείωσης**.
   - Παράδειγμα: στόχος στα 40 m, το drone στα 30 m.
2. Στο **CUE TARGETS** εμφανίζεται ο στόχος (`RFD-1`). Πάτα **GO**: οπλίζει, απογειώνεται και ακολουθεί.
3. Τα κουμπιά στην πτήση είναι στον πίνακα στο τέλος.

---

# 2. ΕΛΚΑΚ (πραγματικοί στόχοι από τον server της ΕΛΚΑΚ)

Χρειάζεσαι: laptop, καλώδιο Ethernet, RFD στο laptop, drone με RFD στο Jetson.

```
ΕΛΚΑΚ server (192.168.1.10) ──Ethernet──► LAPTOP ──RFD──► DRONE
```

### Βήμα 1: Μία φορά, στο γραφείο με internet

```bash
cd ~/Desktop/dev/Interceptor
git pull
sudo usermod -aG dialout $USER     # μετά κάνε log out και log in
```

Σταθερή IP στο Ethernet. Βάλε αυτή που σου έδωσαν, εδώ GCS1 = `192.168.1.101`:

```bash
sudo nmcli con add type ethernet ifname enp130s0 con-name ELKAK \
    ipv4.method manual ipv4.addresses 192.168.1.101/24 ipv6.method disabled
```

### Βήμα 2: Ο χάρτης της ΕΛΚΑΚ (στο γραφείο, με internet)

Εκεί **δεν έχει internet**, οπότε αυτό γίνεται πριν φύγεις.

```bash
python3 src/ground_planner.py --port 9002
```

1. Άνοιξε `http://localhost:9002`.
2. Πήγαινε τον χάρτη **πάνω από την ΕΛΚΑΚ**.
3. Πάτα **«Cache map here»** και περίμενε να γεμίσει.
4. Κλείσε με `Ctrl+C`.

### Βήμα 3: Το drone

1. Το RFD του drone είναι στο **USB του Jetson**.
2. Ανάβεις το drone. Η εφαρμογή ξεκινά μόνη της.

### Βήμα 4: Το laptop στο δίκτυό τους

1. **Κλείσε το Wi-Fi** του laptop (είναι κι αυτό `192.168.1.x` και μπερδεύεται).
2. **Καλώδιο Ethernet** από το laptop στο router τους (Teltonika).
3. Τρέξε:
   ```bash
   sudo nmcli con up ELKAK
   ping -c 3 192.168.1.10
   ```

   Το `ping` πρέπει να απαντά.

### Βήμα 5: Το RFD του laptop

Βάζεις το RFD στο USB του laptop και περιμένεις το **πράσινο φως σταθερό**.

### Βήμα 6: Ξεκινάς

```bash
cd ~/Desktop/dev/Interceptor
python3 src/ground_planner.py --port 9002 \
    --rfd-dev /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG013VP7-if00-port0 \
    --gateway ws://192.168.1.10:18100/gcs --drone-id WASP-01
```

1. Στο terminal πρέπει να δεις `[EKETA] connected to ws://192.168.1.10:18100/gcs`.
2. Άνοιξε `http://localhost:9002`.
3. Πάνω δεξιά πρέπει να δεις **Drone: linked (RFD 0.5s)**.

Από εδώ το πρόγραμμα κάνει μόνο του τα εξής:

- ακούει τους στόχους τους στη θύρα **UDP 18200**,
- απαντά «το πήρα» (ACK) σε κάθε μήνυμα,
- στέλνει τη θέση του drone μας κάθε δευτερόλεπτο, για να μη μας δουν σαν εχθρό.

### Βήμα 7: Πετάς

1. Πάτα την καρτέλα **ELKAK**.
2. Οι στόχοι τους εμφανίζονται στο **CUE TARGETS**.
3. Πάτα **GO** στον στόχο που θες: οπλίζει, απογειώνεται και ακολουθεί 10 m από κάτω.

---

# Κουμπιά στην πτήση (TEST και ΕΛΚΑΚ)

| Θέλω να…                              | Πατάω                                                |
| -------------------------------------------- | --------------------------------------------------------- |
| Ξεκινήσω σε στόχο             | **GO**                                              |
| Αλλάξω στόχο                      | **SWAP** δίπλα στον άλλο στόχο    |
| Νέος στόχος εμφανίστηκε | Ερώτηση «switch?»: **Yes** ή **No** |
| Σταματήσω                           | **ABORT**                                           |
| Γυρίσει σπίτι                    | **RTL**                                             |

- Κάθε εντολή περιμένει απάντηση από το drone μέσω RFD (έως 4 s). Αν δεν έρθει, βγαίνει **κόκκινο μήνυμα**: ξαναπάτα, ή ο πιλότος παίρνει τον έλεγχο.
- Αν χαθεί ο στόχος για 10 s, το drone μένει στη θέση του (ALT_HOLD).

# Αν κάτι δεν πάει καλά

| Βλέπω                   | Κάνω                                                                                            |
| ---------------------------- | --------------------------------------------------------------------------------------------------- |
| `could not open port`      | Το RFD δεν είναι στο USB, ή δεν έκανες `usermod` + log out/in              |
| **RFD LOST**           | Το RFD του drone δεν απαντά: φώτα, Jetson αναμμένο                        |
| Γκρι χάρτης        | Δεν τον αποθήκευσες εκεί (Βήμα 2)                                          |
| `[EKETA] disconnected`     | Καλώδιο, IP (`ip a show enp130s0`), `ping 192.168.1.10`                                  |
| SEEK δεν πατιέται | Κάτω από τα κουμπιά γράφει τι λείπει (GPS, μπαταρία κ.λπ.) |

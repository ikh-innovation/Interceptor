# Ground Planner: TEST και ΕΛΚΑΚ

```
LAPTOP ──RFD──► DRONE (Jetson)      στόχοι + εντολές
LAPTOP ◄─RFD─── DRONE               θέση + κατάσταση του drone
```

- **Εντολές και στόχοι: πάντα από το RFD.**
- **Το drone ακολουθεί τον στόχο συνέχεια, 10 m από κάτω του.** Δεν κάνει ENGAGE, εκτός αν τσεκάρεις «AUTO ENGAGE».
- **Ο πιλότος με τον πομπό μπορεί πάντα να πάρει τον έλεγχο.**

---

# 1. TEST (ψεύτικοι στόχοι που σχεδιάζεις εσύ)

Χρειάζεσαι: laptop με internet, RFD στο laptop, drone με RFD στο Jetson.

### Βήμα 1: Ο κώδικας

Στο laptop:

```bash
cd ~/Desktop/dev/Interceptor
git pull
```

Στο Jetson:

```bash
cd ~/Documents/Interceptor && git pull && sudo systemctl restart wasp-tracker
```

### Βήμα 2: Ανάβεις το drone

1. Το RFD του drone είναι στο **USB του Jetson**.
2. Ανάβεις το drone σε ανοιχτό χώρο, ώστε να πιάσει δορυφόρους.

### Βήμα 3: Το RFD του laptop

Βάζεις το RFD στο USB του laptop και περιμένεις το **πράσινο φως σταθερό**.

### Βήμα 4: Ανοίγεις τον planner

```bash
cd ~/Desktop/dev/Interceptor
python3 src/ground_planner.py --rfd-dev /dev/ttyUSB0 --port 9002 \
    --jetson jetson.34-76-37-31.sslip.io --jetson-http 80
```

Άνοιξε `http://localhost:9002`.

### Βήμα 5: Περιμένεις να βρει δορυφόρους

Πάνω δεξιά, στο **Drone**, βλέπεις π.χ. `linked (RFD 0.5s) · GPS 18 sat`.

1. Μόλις πιάσει GPS, **ο χάρτης πάει μόνος του πάνω στο drone** και το δείχνει.
2. Ο χάρτης της περιοχής **αποθηκεύεται μόνος του** στο laptop, για να φαίνεται και χωρίς internet.
3. Αν το RFD δεν έχει ακουστεί ακόμα, γράφει `(Wi-Fi, RFD not heard yet)`. Το drone φαίνεται μέσω internet, αλλά **εντολές δεν φεύγουν** μέχρι να ακουστεί το RFD.

### Βήμα 6: Πετάς

1. Καρτέλα **TEST**: σχεδίασε στόχο (Circle, Line ή Draw).
   - Το **Altitude** είναι **ύψος πάνω από το σημείο απογείωσης**.
   - Παράδειγμα: στόχος στα 40 m, το drone στα 30 m.
2. Στο **CUE TARGETS** εμφανίζεται ο στόχος (`RFD-1`). Πάτα **GO**: οπλίζει, απογειώνεται και ακολουθεί.

---

# 2. ΕΛΚΑΚ (πραγματικοί στόχοι από τον server της ΕΛΚΑΚ)

Χρειάζεσαι: laptop, καλώδιο Ethernet, RFD στο laptop, drone με RFD στο Jetson.

```
ΕΛΚΑΚ server (192.168.1.10) ──Ethernet──► LAPTOP ──RFD──► DRONE
```

**Εκεί δεν έχει internet.** Το drone και οι στόχοι φαίνονται κανονικά από το RFD. Ο χάρτης όμως φαίνεται μόνο αν έχει αποθηκευτεί από πριν, όσο υπήρχε internet. Αν έχεις πετάξει ήδη στην περιοχή με το TEST, έχει αποθηκευτεί μόνος του.

### Βήμα 1: Μία φορά, στο γραφείο

Σταθερή IP στο Ethernet. Βάλε αυτή που σου έδωσαν, εδώ GCS1 = `192.168.1.101`:

```bash
sudo nmcli con add type ethernet ifname enp130s0 con-name ELKAK \
    ipv4.method manual ipv4.addresses 192.168.1.101/24 ipv6.method disabled
```

### Βήμα 2: Ανάβεις το drone

1. Το RFD του drone είναι στο **USB του Jetson**.
2. Ανάβεις το drone σε ανοιχτό χώρο, ώστε να πιάσει δορυφόρους.

### Βήμα 3: Το laptop στο δίκτυό τους

1. **Κλείσε το Wi-Fi** του laptop (είναι κι αυτό `192.168.1.x` και μπερδεύεται).
2. **Καλώδιο Ethernet** από το laptop στο router τους (Teltonika).
3. Τρέξε:

   ```bash
   sudo nmcli con up ELKAK
   ping -c 3 192.168.1.10
   ```

   Το `ping` πρέπει να απαντά.

### Βήμα 4: Το RFD του laptop

Βάζεις το RFD στο USB του laptop και περιμένεις το **πράσινο φως σταθερό**.

### Βήμα 5: Ανοίγεις τον planner

```bash
cd ~/Desktop/dev/Interceptor
python3 src/ground_planner.py --rfd-dev /dev/ttyUSB0 --port 9002 \
    --gateway ws://192.168.1.10:18100/gcs --drone-id WASP-01
```

1. Στο terminal πρέπει να δεις `[EKETA] connected to ws://192.168.1.10:18100/gcs`.
2. Άνοιξε `http://localhost:9002`.
3. Πάνω δεξιά: `linked (RFD 0.5s) · GPS 18 sat`. Ο χάρτης πάει μόνος του πάνω στο drone.

Από εδώ το πρόγραμμα κάνει μόνο του τα εξής:

- ακούει τους στόχους τους στη θύρα **UDP 18200**,
- απαντά «το πήρα» (ACK) σε κάθε μήνυμα,
- στέλνει τη θέση του drone μας κάθε δευτερόλεπτο, για να μη μας δουν σαν εχθρό.

### Βήμα 6: Πετάς

1. Πάτα την καρτέλα **ELKAK**.
2. Οι στόχοι τους εμφανίζονται στο **CUE TARGETS**.
3. Πάτα **GO** στον στόχο που θες: οπλίζει, απογειώνεται και ακολουθεί 10 m από κάτω.

---

# Κουμπιά στην πτήση (TEST και ΕΛΚΑΚ)

| Θέλω να… | Πατάω |
|---|---|
| Ξεκινήσω σε στόχο | **GO** |
| Αλλάξω στόχο | **SWAP** δίπλα στον άλλο στόχο |
| Νέος στόχος εμφανίστηκε | Ερώτηση «switch?»: **Yes** ή **No** |
| Σταματήσω | **ABORT** |
| Γυρίσει σπίτι | **RTL** |

- Κάθε εντολή περιμένει απάντηση από το drone μέσω RFD (έως 4 s). Αν δεν έρθει, βγαίνει **κόκκινο μήνυμα**: ξαναπάτα, ή ο πιλότος παίρνει τον έλεγχο.
- Αν χαθεί ο στόχος για 10 s, το drone μένει στη θέση του (ALT_HOLD).

# Αν κάτι δεν πάει καλά

| Βλέπω | Κάνω |
|---|---|
| `could not open port` | Το RFD δεν είναι στο USB. Δοκίμασε `ls /dev/ttyUSB*`, ή τρέξε μία φορά `sudo usermod -aG dialout $USER` και κάνε log out/in |
| `RFD LOST` ή `RFD not heard yet` | Το RFD του drone δεν απαντά: φώτα, Jetson αναμμένο, `git pull` στο Jetson |
| `GPS 0 sat` | Το drone δεν έχει πιάσει δορυφόρους ακόμα: ανοιχτός χώρος, περίμενε 1–3 min |
| Γκρι χάρτης | Δεν υπάρχει internet και η περιοχή δεν έχει αποθηκευτεί από πριν |
| `[EKETA] disconnected` | Καλώδιο, IP (`ip a show enp130s0`), `ping 192.168.1.10` |
| SEEK δεν πατιέται | Κάτω από τα κουμπιά γράφει τι λείπει (GPS, μπαταρία κ.λπ.) |

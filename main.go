// stadi -- a GPU slot broker for the box's single 20 GiB card.
//
// Both local engines want the whole card and cannot share it:
//
//   - the llama router keeps a resident rung (17.6-19.7 GiB of 19.98);
//   - stable-diffusion.cpp's DiT wants ~15.8 GiB, and unlike the router it has no
//     way to give it back: sd-server holds its weights until the PROCESS exits.
//
// Neither knows the other exists, so whichever runs first wins and the other OOMs
// mid-load. stadi sits in front of both and owns the slot:
//
//	media request    -> unload the router's models, spawn an sd-server child, proxy
//	chat request     -> stop the child, wait for the VRAM to actually return, proxy
//	metadata request -> proxy untouched (the router's /v1/models loads nothing)
//
// The child is stopped after IDLE with no media request, so the card is never held
// hostage. Because stadi is an OpenRC service it lives outside the sandbox, which
// is what makes starting sd-server possible at all: a sandboxed session cannot run
// rc-service, and sd-cli cannot see /dev/dri. stadi needs neither.
//
// No third-party code: stdlib only, so it builds with no module cache and no network.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type config struct {
	listen    string        // where stadi itself listens
	llm       string        // the llama router
	sd        string        // where the sd child is expected to come up
	idle      time.Duration // kill the child after this long with no media request
	busyWait  time.Duration // how long a chat request waits for a render to finish
	vramWait  time.Duration // how long to wait for VRAM to return
	loadWait  time.Duration // how long to wait for the sd child's port
	freeBelow uint64        // bytes: "the card is free" threshold
	sdBin     string
	sdArgs    []string
	sdLogDir  string
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
		log.Printf("bad duration in %s=%q, using %s", key, v, def)
	}
	return def
}

func loadConfig() config {
	args := strings.Fields(env("STADI_SD_ARGS", ""))
	if len(args) == 0 {
		log.Fatal("STADI_SD_ARGS is empty: refusing to run with no sd-server command line")
	}
	return config{
		listen:    env("STADI_LISTEN", "127.0.0.1:55558"),
		llm:       env("STADI_LLM", "http://127.0.0.1:55555"),
		sd:        env("STADI_SD", "http://127.0.0.1:55557"),
		idle:      envDur("STADI_IDLE", 5*time.Minute),
		busyWait:  envDur("STADI_BUSY_WAIT", 2*time.Minute),
		vramWait:  envDur("STADI_VRAM_WAIT", 45*time.Second),
		loadWait:  envDur("STADI_LOAD_WAIT", 15*time.Minute),
		freeBelow: 2 << 30, // 2 GiB: an idle 7900 XT reads ~0.03 GiB, a resident rung 17+
		sdBin:     env("STADI_SD_BIN", "/usr/bin/sd-server"),
		sdArgs:    args,
		sdLogDir:  env("STADI_SD_LOG_DIR", "/home/user/.dsh/logs"),
	}
}

// vramUsed reads the card's VRAM counter. The box has TWO GPUs (the 7900 XT and a
// GT 1030), and /dev/dri node numbers move between boots, so pick the device with
// the largest total rather than hardcoding a card index.
func vramUsed() (uint64, error) {
	totals, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil || len(totals) == 0 {
		return 0, errors.New("no /sys/class/drm/card*/device/mem_info_vram_total")
	}
	var usedPath string
	var biggest uint64
	for _, p := range totals {
		t, err := readUint(p)
		if err != nil {
			continue
		}
		if t > biggest {
			biggest, usedPath = t, p
		}
	}
	if usedPath == "" {
		return 0, errors.New("no readable VRAM total")
	}
	return readUint(strings.TrimSuffix(usedPath, "_total") + "_used")
}

func readUint(path string) (uint64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
}

func vramString() string {
	used, err := vramUsed()
	if err != nil {
		return "unknown"
	}
	return fmt.Sprintf("%.3f GiB", float64(used)/(1<<30))
}

type broker struct {
	cfg config

	mu    sync.Mutex // serialises slot transitions, never held across a proxy
	child *exec.Cmd
	up    bool
	timer *time.Timer

	media int64 // media requests in flight, so a chat request can wait politely

	sdProxy  *httputil.ReverseProxy
	llmProxy *httputil.ReverseProxy
}

// ---------------------------------------------------------------- slot: sd child

func (b *broker) ensureChild() error {
	if b.up && portAlive(b.cfg.sd) {
		b.touch()
		return nil
	}
	if portAlive(b.cfg.sd) {
		return fmt.Errorf("%s is already served by a process stadi does not own "+
			"(stop the sd service first: rc-service --user sd stop)", b.cfg.sd)
	}
	b.unloadLLM()

	if err := waitVRAMBelow(b.cfg.freeBelow, b.cfg.vramWait); err != nil {
		log.Printf("warning: %v (continuing: the child may still get the card)", err)
	}

	if err := os.MkdirAll(b.cfg.sdLogDir, 0o755); err != nil {
		return fmt.Errorf("log dir: %w", err)
	}
	out, err := os.OpenFile(filepath.Join(b.cfg.sdLogDir, "stadi-sd.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open child stdout: %w", err)
	}
	errf, err := os.OpenFile(filepath.Join(b.cfg.sdLogDir, "stadi-sd.err"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		out.Close()
		return fmt.Errorf("open child stderr: %w", err)
	}

	cmd := exec.Command(b.cfg.sdBin, b.cfg.sdArgs...)
	cmd.Stdout, cmd.Stderr = out, errf
	// One process group, so stopping the child also stops anything it forked.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	log.Printf("starting the sd child: %s %s", b.cfg.sdBin, strings.Join(b.cfg.sdArgs, " "))
	if err := cmd.Start(); err != nil {
		out.Close()
		errf.Close()
		return fmt.Errorf("start sd child: %w", err)
	}
	b.child, b.up = cmd, true
	go func() { // reap, and notice a crash without waiting for a request to fail
		err := cmd.Wait()
		out.Close()
		errf.Close()
		b.mu.Lock()
		if b.child == cmd {
			b.child, b.up = nil, false
		}
		b.mu.Unlock()
		log.Printf("sd child (pid %d) exited: %v", cmd.Process.Pid, err)
	}()

	if err := waitPort(b.cfg.sd, b.cfg.loadWait); err != nil {
		b.stopChild()
		return fmt.Errorf("sd child never came up on %s: %w", b.cfg.sd, err)
	}
	log.Printf("sd child ready on %s (pid %d), vram %s", b.cfg.sd, cmd.Process.Pid, vramString())
	b.touch()
	return nil
}

func (b *broker) stopChild() {
	if b.timer != nil {
		b.timer.Stop()
		b.timer = nil
	}
	if b.child == nil || b.child.Process == nil {
		b.up = false
		return
	}
	pid := b.child.Process.Pid
	log.Printf("stopping the sd child (pid %d), vram %s", pid, vramString())
	// SIGTERM the whole group, then SIGKILL if it will not go. sd-server frees its
	// weights on exit (measured: 12.5 GiB -> 0.03 GiB), which is the entire point.
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if !portAlive(b.cfg.sd) {
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	if portAlive(b.cfg.sd) {
		log.Printf("sd child %d ignored SIGTERM; killing", pid)
		_ = syscall.Kill(-pid, syscall.SIGKILL)
	}
	b.child, b.up = nil, false
	if err := waitVRAMBelow(b.cfg.freeBelow, b.cfg.vramWait); err != nil {
		log.Printf("warning after stopping the child: %v", err)
	} else {
		log.Printf("card released, vram %s", vramString())
	}
}

// touch (re)arms the idle timer: after cfg.idle with no media request the child goes
// away by itself, so the card is never left held.
func (b *broker) touch() {
	if b.timer != nil {
		b.timer.Stop()
	}
	b.timer = time.AfterFunc(b.cfg.idle, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if b.child == nil {
			return
		}
		if atomic.LoadInt64(&b.media) > 0 {
			// A render is still in flight; start the clock again rather than killing it.
			b.touch()
			return
		}
		log.Printf("no media request for %s; releasing the card", b.cfg.idle)
		b.stopChild()
	})
}

// -------------------------------------------------------------------- slot: llama

// unloadLLM asks the router to drop every model it has resident. The router's unload
// is async and force-kills a loading child, so its own answer is the authority.
func (b *broker) unloadLLM() {
	ids, err := routerModels(b.cfg.llm)
	if err != nil {
		log.Printf("warning: cannot list router models: %v", err)
		return
	}
	for _, id := range ids {
		body := strings.NewReader(`{"model":` + strconv.Quote(id) + `}`)
		req, err := http.NewRequest(http.MethodPost, b.cfg.llm+"/models/unload", body)
		if err != nil {
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Printf("unload %s: %v", id, err)
			continue
		}
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		log.Printf("unload %s -> %s %s", id, resp.Status, strings.TrimSpace(string(msg)))
	}
}

func routerModels(base string) ([]string, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(base + "/v1/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(payload.Data))
	for _, m := range payload.Data {
		ids = append(ids, m.ID)
	}
	return ids, nil
}

// ------------------------------------------------------------------------ helpers

func portAlive(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	c, err := net.DialTimeout("tcp", u.Host, 750*time.Millisecond)
	if err != nil {
		return false
	}
	c.Close()
	return true
}

func waitPort(base string, limit time.Duration) error {
	deadline := time.Now().Add(limit)
	for time.Now().Before(deadline) {
		if portAlive(base) {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out after %s", limit)
}

func waitVRAMBelow(limit uint64, howLong time.Duration) error {
	deadline := time.Now().Add(howLong)
	var last uint64
	for time.Now().Before(deadline) {
		used, err := vramUsed()
		if err == nil {
			last = used
			if used < limit {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("vram still %.3f GiB after %s (wanted below %.3f GiB)",
		float64(last)/(1<<30), howLong, float64(limit)/(1<<30))
}

// ---------------------------------------------------------------- routing + proxy

// Generation paths actually LOAD the model, so they may spawn the child and they refresh
// the idle clock. Observation paths only ever REPORT on it: /sdapi/v1/progress, /options,
// /sd-models, /samplers and friends. Keeping the two apart matters, because a monitoring
// loop polling /progress would otherwise spawn the child and pin the card forever -- the
// exact thing the idle timer exists to undo.
//
// The upshot is that "is sd loaded?" needs no state of its own: it is simply whether the
// child is running, which /status reports as sd_child_up.
func isGeneration(p string) bool {
	if strings.HasPrefix(p, "/v1/images/") {
		return true
	}
	switch p {
	case "/sdapi/v1/txt2img", "/sdapi/v1/img2img":
		return true
	}
	return false
}

func isObservation(p string) bool { return strings.HasPrefix(p, "/sdapi/") }

// isMetadata is proxied WITHOUT a transition. The picker polls /v1/models constantly,
// and the router answers it without loading anything, so it must never evict the
// child -- otherwise every UI refresh would kill a render.
func isMetadata(p string) bool {
	switch p {
	case "/v1/models", "/props", "/health", "/healthz":
		return true
	}
	return false
}

func reverseProxy(b *broker, base string) *httputil.ReverseProxy {
	target, err := url.Parse(base)
	if err != nil {
		log.Fatalf("bad target %q: %v", base, err)
	}
	p := httputil.NewSingleHostReverseProxy(target)
	// -1 flushes every write straight through: chat is SSE, and sd streams
	// image_generation.* events.
	p.FlushInterval = -1
	p.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("%s %s -> %s: %v", r.Method, r.URL.Path, base, err)
		http.Error(w, "stadi: upstream failed: "+err.Error(), http.StatusBadGateway)
	}
	return p
}

func (b *broker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/status":
		b.status(w)
	case r.URL.Path == "/release":
		b.mu.Lock()
		b.stopChild()
		b.mu.Unlock()
		fmt.Fprintln(w, "released")
	case isGeneration(r.URL.Path):
		b.mu.Lock()
		err := b.ensureChild()
		b.mu.Unlock()
		if err != nil {
			log.Printf("generation request refused: %v", err)
			http.Error(w, "stadi: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		atomic.AddInt64(&b.media, 1)
		b.sdProxy.ServeHTTP(w, r)
		atomic.AddInt64(&b.media, -1)
		// Re-arm from the END of the request. ensureChild() touches BEFORE the render, and a
		// cold render takes ~8-10 min, so a 5-minute timer armed there would kill the child
		// mid-load. The count above is what tells the timer a render is still going.
		b.mu.Lock()
		b.touch()
		b.mu.Unlock()
	case isObservation(r.URL.Path):
		// Never spawn for a read: answer honestly instead, so a monitor cannot hold the card.
		b.mu.Lock()
		up := b.up && portAlive(b.cfg.sd)
		b.mu.Unlock()
		if !up {
			http.Error(w, "stadi: sd is not loaded (no child running; see /status)", http.StatusServiceUnavailable)
			return
		}
		b.sdProxy.ServeHTTP(w, r)
	case isMetadata(r.URL.Path):
		b.llmProxy.ServeHTTP(w, r)
	default:
		// A chat request wants the card. If a render is in flight, wait a bounded
		// while for it to finish -- then say so plainly instead of killing it.
		if n := atomic.LoadInt64(&b.media); n > 0 {
			deadline := time.Now().Add(b.cfg.busyWait)
			for atomic.LoadInt64(&b.media) > 0 && time.Now().Before(deadline) {
				time.Sleep(500 * time.Millisecond)
			}
			if atomic.LoadInt64(&b.media) > 0 {
				http.Error(w, "stadi: an image generation is in progress on the card; retry shortly",
					http.StatusServiceUnavailable)
				return
			}
		}
		b.mu.Lock()
		b.stopChild()
		b.mu.Unlock()
		b.llmProxy.ServeHTTP(w, r)
	}
}

func (b *broker) status(w http.ResponseWriter) {
	b.mu.Lock()
	pid := 0
	if b.child != nil && b.child.Process != nil {
		pid = b.child.Process.Pid
	}
	up := b.up
	b.mu.Unlock()
	used, _ := vramUsed()
	out := map[string]any{
		"slot":            slotName(up),
		"sd_child_up":     up,
		"sd_child_pid":    pid,
		"sd_listen":       b.cfg.sd,
		"llm":             b.cfg.llm,
		"media_in_flight": atomic.LoadInt64(&b.media),
		"idle_timeout_s":  b.cfg.idle.Seconds(),
		"vram_used_gib":   float64(used) / (1 << 30),
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

func slotName(sdUp bool) string {
	if sdUp {
		return "sd"
	}
	return "llm"
}

// ---------------------------------------------------------------------------- main

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("stadi: ")
	cfg := loadConfig()

	b := &broker{cfg: cfg}
	b.sdProxy = reverseProxy(b, cfg.sd)
	b.llmProxy = reverseProxy(b, cfg.llm)

	log.Printf("listening on %s | llm %s | sd %s | idle %s | vram %s",
		cfg.listen, cfg.llm, cfg.sd, cfg.idle, vramString())

	// supervise-daemon stops us with SIGTERM. Because the child sits in its own process
	// group (Setpgid) it would NOT die with us, and an orphaned sd-server is exactly the
	// 12.5 GiB hostage this program exists to prevent -- so take it down on the way out.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-stop
		log.Printf("signal received; stopping the sd child and exiting")
		b.mu.Lock()
		b.stopChild()
		b.mu.Unlock()
		os.Exit(0)
	}()

	srv := &http.Server{
		Addr:              cfg.listen,
		Handler:           b,
		ReadHeaderTimeout: 15 * time.Second,
		// No WriteTimeout: an image generation can legitimately take ten minutes,
		// and a default would guillotine it.
	}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

package decision

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

// helm2_prompts.json comes from llama/helm2/golden_prompts.py, which builds
// each request's model input with the saina package and Helm 2's chat template.
func TestHelm2MatchesSaina(t *testing.T) {
	data, err := os.ReadFile("testdata/helm2_prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Request  string   `json:"request"`
		Segments []string `json:"segments"`
		Counts   []int    `json:"counts"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for i, tc := range cases {
		var req Request
		if err := json.Unmarshal([]byte(tc.Request), &req); err != nil {
			t.Fatal(err)
		}
		c, err := Compile(req, "helm2")
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if len(c.Request.Segments) != len(tc.Segments) {
			t.Fatalf("case %d: %d segments, want %d", i, len(c.Request.Segments), len(tc.Segments))
		}
		for j := range tc.Segments {
			if got, want := c.Request.Segments[j], tc.Segments[j]; got != want {
				n := 0
				for n < len(got) && n < len(want) && got[n] == want[n] {
					n++
				}
				t.Fatalf("case %d segment %d differs at byte %d:\n got %q\nwant %q", i, j, n,
					got[max(0, n-40):min(len(got), n+40)], want[max(0, n-40):min(len(want), n+40)])
			}
		}
		for j, f := range c.Request.Fields {
			if f.Count != tc.Counts[j] || f.Question != [2]int{j + 1, j + 2} || len(f.Options) != 0 {
				t.Fatalf("case %d field %d: %+v", i, j, f)
			}
		}
	}
}

func TestHelm2Answer(t *testing.T) {
	req := Request{Model: "helm2", State: json.RawMessage(`"I was charged twice."`), Questions: &Questions{}}
	req.Questions.Set("refund", Question{Type: "noul", Instructions: json.RawMessage(`"Refund?"`)})
	req.Questions.Set("team", Question{Type: "choice", Instructions: json.RawMessage(`"Team?"`),
		Criteria: json.RawMessage(`{"billing": null, "technical": null, "shipping": null}`)})
	req.Questions.Set("severity", Question{Type: "score", Instructions: json.RawMessage(`"Severity?"`),
		Criteria: json.RawMessage(`["low", "high"]`)})
	c, err := Compile(req, "helm2")
	if err != nil {
		t.Fatal(err)
	}
	// Yes is read first: a high first logit means true.
	got, err := c.Answer("helm2", llm.ScoreResponse{Logits: [][]float32{{4, 0}, {0, 5, 0}, {0, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	refund, _ := got.Answers.Get("refund")
	if p := refund.(NoulAnswer).Noul; p < 0.98 {
		t.Fatalf("noul = %v", p)
	}
	team, _ := got.Answers.Get("team")
	if team.(ChoiceAnswer).Choice != "technical" {
		t.Fatalf("choice = %+v", team)
	}
	severity, _ := got.Answers.Get("severity")
	if s := severity.(ScoreAnswer).Score; s != 0.5 {
		t.Fatalf("score = %v", s)
	}
}

func TestHelm2Limits(t *testing.T) {
	keys := func(n int) json.RawMessage {
		parts := make([]string, n)
		for i := range parts {
			parts[i] = `"k` + strings.Repeat("x", i) + `": null`
		}
		return json.RawMessage("{" + strings.Join(parts, ",") + "}")
	}
	for _, tc := range []struct {
		name string
		q    Question
		ok   bool
	}{
		{"255 options", Question{Type: "choice", Instructions: json.RawMessage(`"q"`), Criteria: keys(255)}, true},
		{"256 options", Question{Type: "choice", Instructions: json.RawMessage(`"q"`), Criteria: keys(256)}, false},
		{"one option", Question{Type: "choice", Instructions: json.RawMessage(`"q"`), Criteria: keys(1)}, false},
		{"blank key", Question{Type: "choice", Instructions: json.RawMessage(`"q"`), Criteria: json.RawMessage(`{" ": null, "b": null}`)}, false},
		{"number description", Question{Type: "choice", Instructions: json.RawMessage(`"q"`), Criteria: json.RawMessage(`{"a": 1, "b": null}`)}, false},
		{"11 levels", Question{Type: "score", Instructions: json.RawMessage(`"q"`), Criteria: json.RawMessage(`["0","1","2","3","4","5","6","7","8","9","10"]`)}, false},
		{"noul extra key", Question{Type: "noul", Instructions: json.RawMessage(`"q"`), Criteria: json.RawMessage(`{"maybe": "x"}`)}, false},
		{"empty instructions", Question{Type: "noul", Instructions: json.RawMessage(`""`)}, false},
		{"number instructions", Question{Type: "noul", Instructions: json.RawMessage(`3`)}, false},
		{"unknown type", Question{Type: "multi", Instructions: json.RawMessage(`"q"`)}, false},
	} {
		req := Request{Model: "helm2", State: json.RawMessage(`"s"`), Questions: &Questions{}}
		req.Questions.Set("q", tc.q)
		_, err := Compile(req, "helm2")
		if (err == nil) != tc.ok {
			t.Errorf("%s: err = %v", tc.name, err)
		}
	}
	req := Request{Model: "helm2", State: json.RawMessage(`1e400`), Questions: &Questions{}}
	req.Questions.Set("q", Question{Type: "noul", Instructions: json.RawMessage(`"q"`)})
	if _, err := Compile(req, "helm2"); err == nil {
		t.Error("non-string state accepted")
	}
	req.State = json.RawMessage(`{"x": 1e400}`)
	if _, err := Compile(req, "helm2"); err == nil {
		t.Error("infinite number accepted")
	}
	req.State = json.RawMessage(`"s"`)
	req.Images = append(req.Images, []byte{1})
	if _, err := Compile(req, "helm2"); err == nil {
		t.Error("image accepted")
	}
}

func TestHelmCodes(t *testing.T) {
	// The answer codes are part of the model contract: 26 letters, then the
	// two-letter codes that are single tokens, 588 in all.
	if len(helmCodes) != 588 || helmCodes[0] != "A" || helmCodes[26] != "AA" || helmCodes[587] != "ZZ" {
		t.Fatalf("unexpected answer codes: %d", len(helmCodes))
	}
	if len(slices.Compact(slices.Sorted(slices.Values(helmCodes)))) != 588 {
		t.Fatal("duplicate answer codes")
	}
}

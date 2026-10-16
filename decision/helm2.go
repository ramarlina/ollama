package decision

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/llm"
)

// Saina Helm 2 answers every question from one pass over a shared prompt: a
// fixed instruction, then the context and all questions as JSON, followed by
// one readout slot per question ("\nDecision q0:", "\nDecision q1:", ...). The
// runner applies Helm 2's routed head to the final hidden state at each slot.
// The prompt follows saina.prompt.encode_shared_prompt and the option texts
// follow saina.prepare.prepare_systemone (Apache-2.0), byte for byte.
const helm2Instruction = "Use the shared context to answer every question. For single_label choose one option; " +
	"for multi_label assess each option independently.\n"

// helm2Chat is the Qwen 3.5 chat template rendered for one user message with
// thinking disabled, as Helm 2 was trained.
const (
	helm2ChatStart = "<|im_start|>user\n"
	helm2ChatEnd   = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

func encodeHelm2(req Request, c *Compiled) error {
	state, err := helm2Content(req.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	var items []pyValue
	var fields []llm.ScoreField
	for name, q := range req.Questions.All() {
		f, texts, err := helm2Field(name, q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		options := make([]pyValue, len(texts))
		for j, text := range texts {
			options[j] = pyObject{{"code", f.Choices[j].Code}, {"text", text}}
		}
		i := len(c.fields)
		items = append(items, pyObject{
			{"id", "q" + strconv.Itoa(i)},
			{"question", f.Description},
			{"mode", "single_label"},
			{"options", options},
		})
		fieldType := 0
		switch q.Type {
		case "choice":
			fieldType = 1
		case "score":
			fieldType = 2
		}
		// Segment 0 is the prompt; segment i+1 is question i's readout slot.
		fields = append(fields, llm.ScoreField{Type: fieldType, Question: [2]int{i + 1, i + 2}, Count: len(texts)})
		c.fields = append(c.fields, f)
	}
	content, err := pyDumps(pyObject{{"context", state}, {"questions", items}})
	if err != nil {
		return err
	}
	// The chat template trims the message; the content already starts and ends
	// without whitespace.
	segments := []string{helm2ChatStart + helm2Instruction + content + helm2ChatEnd}
	images := len(req.Images) > 0
	if images {
		// Helm 2 Omni reads each image as <|vision_start|> image <|vision_end|> at the start of the user turn,
		// before the text, as saina.prompt.insert_images places them: the images go in at the boundary after the
		// chat start, and every readout slot moves one segment later.
		segments = []string{helm2ChatStart, helm2Instruction + content + helm2ChatEnd}
		for i := range fields {
			fields[i].Question = [2]int{fields[i].Question[0] + 1, fields[i].Question[1] + 1}
		}
	}
	for i := range fields {
		// Each slot is tokenized on its own, as in training.
		segments = append(segments, fmt.Sprintf("\nDecision q%d:", i))
	}
	c.Request = llm.ScoreRequest{Segments: segments, Fields: fields}
	if images {
		c.Request.Images = req.Images
		c.Request.ImagePosition = 1
	}
	return nil
}

// helm2Field compiles a question into coded choices and the option texts Helm 2
// reads: "Yes: …"/"No: …" for noul, "key" or "key: description" for choice,
// and "index" or "index: description" for score.
func helm2Field(name string, q Question) (compiledField, []string, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, nil, fmt.Errorf("field name must not be empty")
	}
	var err error
	f.Description, err = helm2Content(q.Instructions)
	if err != nil || strings.TrimSpace(f.Description) == "" {
		return f, nil, fmt.Errorf("instructions must be a nonempty string, object, or array")
	}
	var criteria pyValue
	if raw := bytes.TrimSpace(q.Criteria); len(raw) > 0 && string(raw) != "null" {
		criteria, err = pyLoads(raw)
		if err != nil {
			return f, nil, fmt.Errorf("criteria: %w", err)
		}
	}
	var texts []string
	add := func(value any, description, text string) {
		f.Choices = append(f.Choices, Choice{Code: helmCodes[len(f.Choices)], Value: value, Description: description})
		texts = append(texts, text)
	}
	// labelled renders "label" or "label: description" like saina's _labelled.
	labelled := func(label string, value pyValue) (string, string, error) {
		if value == nil {
			return label, label, nil
		}
		description, err := pyRender(value)
		if err != nil {
			return "", "", err
		}
		return description, label + ": " + description, nil
	}
	limit := maxHelmChoices
	switch q.Type {
	case "noul":
		yes, no := pyValue("The answer is yes"), pyValue("The answer is no")
		if criteria != nil {
			values, ok := criteria.(pyObject)
			if !ok {
				return f, nil, fmt.Errorf("noul criteria must be an object of true/false descriptions")
			}
			for _, member := range values {
				switch member.key {
				case "true":
					yes = member.value
				case "false":
					no = member.value
				default:
					return f, nil, fmt.Errorf("noul criteria must contain only true/false descriptions")
				}
			}
		}
		texts := make([]string, 2)
		for i, value := range []pyValue{yes, no} {
			// A null description renders as "null", as in saina.
			if value != nil && !pyDescription(value) {
				return f, nil, fmt.Errorf("noul descriptions must be strings, objects, arrays, or null")
			}
			if texts[i], err = pyRender(value); err != nil {
				return f, nil, err
			}
		}
		// Helm reads yes first; Answer finds the true choice by value.
		add(true, texts[0], "Yes: "+texts[0])
		add(false, texts[1], "No: "+texts[1])
	case "choice":
		values, ok := criteria.(pyObject)
		if !ok {
			return f, nil, fmt.Errorf("choice criteria must map option keys to descriptions or null")
		}
		for _, member := range values {
			if strings.TrimSpace(member.key) == "" {
				return f, nil, fmt.Errorf("choice keys must not be empty")
			}
			if member.value != nil && !pyDescription(member.value) {
				return f, nil, fmt.Errorf("choice descriptions must be strings, objects, arrays, or null")
			}
			if len(f.Choices) == limit {
				return f, nil, fmt.Errorf("criteria must contain 2–%d candidates", limit)
			}
			description, text, err := labelled(member.key, member.value)
			if err != nil {
				return f, nil, err
			}
			add(member.key, description, text)
		}
	case "score":
		limit = 10
		values, ok := criteria.([]pyValue)
		if !ok {
			return f, nil, fmt.Errorf("score criteria must be an array of descriptions")
		}
		if len(values) > limit {
			return f, nil, fmt.Errorf("criteria must contain 2–%d candidates", limit)
		}
		for i, value := range values {
			if value != nil && !pyDescription(value) {
				return f, nil, fmt.Errorf("score descriptions must be strings, objects, arrays, or null")
			}
			label := strconv.Itoa(i)
			description, text, err := labelled(label, value)
			if err != nil {
				return f, nil, err
			}
			add(label, description, text)
		}
	default:
		return f, nil, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(f.Choices) < 2 || len(f.Choices) > limit {
		return f, nil, fmt.Errorf("criteria must contain 2–%d candidates", limit)
	}
	return f, texts, nil
}

// helm2Content renders request content as saina's render does: strings as
// they are, objects and arrays as json.dumps(value, ensure_ascii=False).
func helm2Content(raw json.RawMessage) (string, error) {
	value, err := pyLoads(raw)
	if err != nil {
		return "", err
	}
	if !pyDescription(value) {
		return "", fmt.Errorf("must be a string, object, or array")
	}
	return pyRender(value)
}

// pyValue is a JSON value as Python's json.loads returns it: string,
// json.Number, bool, nil, []pyValue, or pyObject, which keeps key order.
type pyValue = any

type pyMember struct {
	key   string
	value pyValue
}

type pyObject []pyMember

func pyDescription(value pyValue) bool {
	switch value.(type) {
	case string, pyObject, []pyValue:
		return true
	}
	return false
}

func pyRender(value pyValue) (string, error) {
	if s, ok := value.(string); ok {
		return s, nil
	}
	return pyDumps(value)
}

// pyLoads decodes JSON the way Python's json.loads does into a dict: key
// order is kept, and a repeated key keeps its first position and last value.
func pyLoads(raw json.RawMessage) (pyValue, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	value, err := pyDecode(dec)
	if err != nil {
		return nil, fmt.Errorf("must be a JSON value: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("must be a single JSON value")
	}
	return value, nil
}

func pyDecode(dec *json.Decoder) (pyValue, error) {
	token, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := token.(type) {
	case json.Delim:
		switch t {
		case '[':
			values := []pyValue{}
			for dec.More() {
				value, err := pyDecode(dec)
				if err != nil {
					return nil, err
				}
				values = append(values, value)
			}
			_, err := dec.Token()
			return values, err
		case '{':
			object := pyObject{}
			index := map[string]int{}
			for dec.More() {
				key, err := dec.Token()
				if err != nil {
					return nil, err
				}
				name, ok := key.(string)
				if !ok {
					return nil, fmt.Errorf("invalid object key")
				}
				value, err := pyDecode(dec)
				if err != nil {
					return nil, err
				}
				if i, ok := index[name]; ok {
					object[i].value = value
				} else {
					index[name] = len(object)
					object = append(object, pyMember{name, value})
				}
			}
			_, err := dec.Token()
			return object, err
		}
		return nil, fmt.Errorf("unexpected delimiter %v", t)
	default:
		return t, nil
	}
}

// pyDumps matches Python's json.dumps(value, ensure_ascii=False) with its
// default separators and its integer/float rendering. Non-finite numbers are
// rejected, as with allow_nan=False.
func pyDumps(value pyValue) (string, error) {
	var b strings.Builder
	if err := pyWrite(&b, value); err != nil {
		return "", err
	}
	return b.String(), nil
}

func pyWrite(b *strings.Builder, value pyValue) error {
	switch v := value.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case string:
		pyString(b, v)
	case json.Number:
		text, err := pyNumber(v)
		if err != nil {
			return err
		}
		b.WriteString(text)
	case []pyValue:
		b.WriteByte('[')
		for i, child := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			if err := pyWrite(b, child); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case pyObject:
		b.WriteByte('{')
		for i, member := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			pyString(b, member.key)
			b.WriteString(": ")
			if err := pyWrite(b, member.value); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("unsupported JSON value %T", value)
	}
	return nil
}

func pyString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pyNumber renders a JSON number as Python does after json.loads: integers
// exactly, other numbers as float repr.
func pyNumber(v json.Number) (string, error) {
	s := string(v)
	if !strings.ContainsAny(s, ".eE") {
		s = strings.TrimPrefix(s, "-")
		if s == "0" {
			return "0", nil
		}
		return string(v), nil
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsInf(n, 0) || math.IsNaN(n) {
		return "", fmt.Errorf("numbers must be finite")
	}
	if a := math.Abs(n); a >= 1e16 || (a != 0 && a < 1e-4) {
		return strconv.FormatFloat(n, 'e', -1, 64), nil
	}
	s = strconv.FormatFloat(n, 'f', -1, 64)
	if !strings.ContainsRune(s, '.') {
		s += ".0"
	}
	return s, nil
}

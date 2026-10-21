// Resolve compiled openDAW's bundler-style relative imports in this Node harness.
import {registerHooks} from "node:module"
import {pathToFileURL} from "node:url"

const packageRoot = pathToFileURL(process.argv[2] + "/packages/").href
registerHooks({
    resolve(specifier, context, nextResolve) {
        try {
            return nextResolve(specifier, context)
        } catch (error) {
            if (!["ERR_MODULE_NOT_FOUND", "ERR_UNSUPPORTED_DIR_IMPORT"].includes(error.code)
                || !context.parentURL?.startsWith(packageRoot)
                || !(specifier === "." || specifier === ".."
                    || specifier.startsWith("./") || specifier.startsWith("../"))) throw error
            const suffixes = specifier === "." || specifier === ".." ? ["/index.js"] : [".js", "/index.js"]
            for (const suffix of suffixes) {
                try {
                    return nextResolve(specifier + suffix, context)
                } catch (candidateError) {
                    if (!["ERR_MODULE_NOT_FOUND", "ERR_UNSUPPORTED_DIR_IMPORT"].includes(candidateError.code)) throw candidateError
                }
            }
            throw error
        }
    }
})
